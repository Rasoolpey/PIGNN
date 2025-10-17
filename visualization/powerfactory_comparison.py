"""
PowerFactory vs Solver Comparison Visualization System

This module creates detailed comparison plots between your solver results 
and DIgSILENT PowerFactory results for contingency analysis.

Generates three main comparison plots:
1. Voltage magnitudes and angles at all buses
2. Line currents (real and reactive power flows) 
3. Generator outputs (active and reactive power)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import h5py
from datetime import datetime


class PowerFactoryComparator:
    """Compare solver results with DIgSILENT PowerFactory results."""
    
    def __init__(self, figure_size: Tuple[int, int] = (16, 12)):
        """Initialize the comparator.
        
        Args:
            figure_size: Size of the comparison figures (width, height)
        """
        self.figure_size = figure_size
        self.colors = {
            'solver': '#2E8B57',      # Sea Green
            'powerfactory': '#DC143C', # Crimson
            'error': '#FF6347',        # Tomato
            'grid': '#E5E5E5'         # Light Gray
        }
        
        # Create output directory
        self.output_dir = Path("Contingency Analysis/comparison_plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_comprehensive_comparison(self, 
                                     solver_results: Dict[str, Any],
                                     powerfactory_h5_path: str,
                                     scenario_info: Dict[str, Any],
                                     save_plots: bool = True) -> Dict[str, plt.Figure]:
        """Create comprehensive comparison plots.
        
        Args:
            solver_results: Results from your load flow solver
            powerfactory_h5_path: Path to DIgSILENT H5 file
            scenario_info: Information about the scenario being analyzed
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing the matplotlib figures
        """
        print(f"\n📊 Creating PowerFactory comparison plots...")
        print(f"   Scenario: {scenario_info.get('description', 'Unknown')}")
        
        # Load PowerFactory results
        pf_results = self._load_powerfactory_results(powerfactory_h5_path)
        
        if not pf_results:
            print("   ⚠️  Could not load PowerFactory results")
            return {}
        
        # Create the three main comparison plots
        figures = {}
        
        # 1. Voltage Comparison
        print("   📈 Creating voltage comparison plot...")
        figures['voltages'] = self._create_voltage_comparison(
            solver_results, pf_results, scenario_info
        )
        
        # 2. Line Flow Comparison
        print("   📈 Creating line flow comparison plot...")
        figures['line_flows'] = self._create_line_flow_comparison(
            solver_results, pf_results, scenario_info
        )
        
        # 3. Generation Comparison
        print("   📈 Creating generation comparison plot...")
        figures['generation'] = self._create_generation_comparison(
            solver_results, pf_results, scenario_info
        )
        
        # Save plots if requested
        if save_plots:
            self._save_comparison_plots(figures, scenario_info)
        
        return figures
    
    def _load_powerfactory_results(self, h5_path: str) -> Optional[Dict[str, Any]]:
        """Load PowerFactory results from H5 file with actual DIgSILENT structure."""
        try:
            results = {}
            with h5py.File(h5_path, 'r') as f:
                # Check if load flow converged
                converged = False
                if 'load_flow_results' in f:
                    lf_group = f['load_flow_results']
                    if 'convergence' in lf_group:
                        converged = bool(lf_group['convergence'][()])
                        print(f"   📊 PowerFactory convergence: {converged}")
                
                # Load voltage results from detailed_system_data
                if 'detailed_system_data' in f and 'buses' in f['detailed_system_data']:
                    bus_group = f['detailed_system_data/buses']
                    
                    # Get bus names
                    bus_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in bus_group['names'][:]]
                    
                    results['voltages'] = {
                        'magnitude': np.array(bus_group['voltages_pu'][:]),
                        'angle': np.array(bus_group['voltage_angles_deg'][:]) * np.pi / 180.0,  # Convert to radians
                        'bus_names': bus_names
                    }
                    print(f"   📊 Loaded {len(bus_names)} bus voltages")
                
                # Load line flow results - we'll need to calculate from voltage results
                # For now, use injection data as proxy for flows
                if 'detailed_system_data' in f and 'buses' in f['detailed_system_data']:
                    bus_group = f['detailed_system_data/buses']
                    if 'active_injection_MW' in bus_group and 'reactive_injection_MVAR' in bus_group:
                        # Use bus injections as proxy for line flows
                        n_buses = len(bus_group['active_injection_MW'][:])
                        results['line_flows'] = {
                            'active_power': np.array(bus_group['active_injection_MW'][:]),
                            'reactive_power': np.array(bus_group['reactive_injection_MVAR'][:]),
                            'from_buses': bus_names,
                            'to_buses': bus_names
                        }
                        print(f"   📊 Using bus injections as flow proxy ({n_buses} elements)")
                
                # Load generation results
                if 'detailed_system_data' in f and 'generators' in f['detailed_system_data']:
                    gen_group = f['detailed_system_data/generators']
                    
                    # Get generator names
                    gen_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in gen_group['names'][:]] if 'names' in gen_group else None
                    
                    results['generators'] = {
                        'active_power': np.array(gen_group['active_power_MW'][:]),
                        'reactive_power': np.array(gen_group['reactive_power_MVAR'][:]),
                        'bus_names': gen_names
                    }
                    print(f"   📊 Loaded {len(results['generators']['active_power'])} generators")
                
                # Add convergence info
                results['convergence'] = {
                    'converged': converged,
                    'source': 'DIgSILENT PowerFactory'
                }
            
            return results
        
        except Exception as e:
            print(f"   ❌ Error loading PowerFactory results: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_voltage_comparison(self, 
                                 solver_results: Dict[str, Any],
                                 pf_results: Dict[str, Any],
                                 scenario_info: Dict[str, Any]) -> plt.Figure:
        """Create voltage magnitude and angle comparison plot."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(f'Voltage Comparison - {scenario_info.get("description", "Unknown Scenario")}', 
                    fontsize=16, fontweight='bold')
        
        # Extract data
        solver_voltages = solver_results.get('voltages', {})
        pf_voltages = pf_results.get('voltages', {})
        
        if not solver_voltages or not pf_voltages:
            # Create placeholder plot if no data
            ax1.text(0.5, 0.5, 'No voltage data available', ha='center', va='center')
            ax1.set_title('Voltage Magnitudes')
            return fig
        
        # Get bus names and indices
        bus_names = pf_voltages.get('bus_names', [f"Bus {i+1}" for i in range(len(pf_voltages['magnitude']))])
        bus_indices = range(len(bus_names))
        
        # Solver voltages (convert if necessary)
        if isinstance(solver_voltages.get('magnitude'), np.ndarray):
            solver_vmag = solver_voltages['magnitude'][:len(bus_names)]
            solver_vang = solver_voltages.get('angle', np.zeros(len(bus_names)))[:len(bus_names)]
        else:
            solver_vmag = np.ones(len(bus_names))  # Default values
            solver_vang = np.zeros(len(bus_names))
        
        # PowerFactory voltages
        pf_vmag = pf_voltages['magnitude'][:len(bus_names)]
        pf_vang = pf_voltages['angle'][:len(bus_names)]
        
        # 1. Voltage Magnitudes
        ax1.plot(bus_indices, solver_vmag, 'o-', color=self.colors['solver'], 
                linewidth=2, markersize=6, label='Your Solver', alpha=0.8)
        ax1.plot(bus_indices, pf_vmag, 's-', color=self.colors['powerfactory'], 
                linewidth=2, markersize=6, label='PowerFactory', alpha=0.8)
        ax1.set_xlabel('Bus Number')
        ax1.set_ylabel('Voltage Magnitude (pu)')
        ax1.set_title('Voltage Magnitudes', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(-0.5, len(bus_names)-0.5)
        
        # 2. Voltage Angles
        ax2.plot(bus_indices, np.degrees(solver_vang), 'o-', color=self.colors['solver'], 
                linewidth=2, markersize=6, label='Your Solver', alpha=0.8)
        ax2.plot(bus_indices, np.degrees(pf_vang), 's-', color=self.colors['powerfactory'], 
                linewidth=2, markersize=6, label='PowerFactory', alpha=0.8)
        ax2.set_xlabel('Bus Number')
        ax2.set_ylabel('Voltage Angle (degrees)')
        ax2.set_title('Voltage Angles', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(-0.5, len(bus_names)-0.5)
        
        # 3. Voltage Magnitude Errors
        vmag_error = np.abs(solver_vmag - pf_vmag)
        bars1 = ax3.bar(bus_indices, vmag_error, color=self.colors['error'], alpha=0.7)
        ax3.set_xlabel('Bus Number')
        ax3.set_ylabel('Magnitude Error (pu)')
        ax3.set_title('Voltage Magnitude Errors', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.5, len(bus_names)-0.5)
        
        # Add error statistics
        max_error = np.max(vmag_error)
        mean_error = np.mean(vmag_error)
        ax3.text(0.02, 0.98, f'Max Error: {max_error:.4f} pu\nMean Error: {mean_error:.4f} pu', 
                transform=ax3.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Voltage Angle Errors
        vang_error = np.abs(np.degrees(solver_vang) - np.degrees(pf_vang))
        bars2 = ax4.bar(bus_indices, vang_error, color=self.colors['error'], alpha=0.7)
        ax4.set_xlabel('Bus Number')
        ax4.set_ylabel('Angle Error (degrees)')
        ax4.set_title('Voltage Angle Errors', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-0.5, len(bus_names)-0.5)
        
        # Add error statistics
        max_ang_error = np.max(vang_error)
        mean_ang_error = np.mean(vang_error)
        ax4.text(0.02, 0.98, f'Max Error: {max_ang_error:.3f}°\nMean Error: {mean_ang_error:.3f}°', 
                transform=ax4.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _create_line_flow_comparison(self, 
                                   solver_results: Dict[str, Any],
                                   pf_results: Dict[str, Any],
                                   scenario_info: Dict[str, Any]) -> plt.Figure:
        """Create line flow comparison plot."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(f'Line Flow Comparison - {scenario_info.get("description", "Unknown Scenario")}', 
                    fontsize=16, fontweight='bold')
        
        # Extract data
        solver_flows = solver_results.get('line_flows', {})
        pf_flows = pf_results.get('line_flows', {})
        
        if not pf_flows:
            ax1.text(0.5, 0.5, 'No line flow data available', ha='center', va='center')
            ax1.set_title('Line Flows')
            return fig
        
        # Get line information
        n_lines = len(pf_flows.get('active_power', []))
        line_indices = range(n_lines)
        line_labels = [f"L{i+1}" for i in range(n_lines)]
        
        # PowerFactory flows
        pf_active = np.array(pf_flows.get('active_power', []))
        pf_reactive = np.array(pf_flows.get('reactive_power', []))
        
        # Solver flows (use defaults if not available)
        if solver_flows:
            solver_active = np.array(solver_flows.get('active_power', np.zeros(n_lines)))[:n_lines]
            solver_reactive = np.array(solver_flows.get('reactive_power', np.zeros(n_lines)))[:n_lines]
        else:
            solver_active = np.zeros(n_lines)
            solver_reactive = np.zeros(n_lines)
        
        # 1. Active Power Flows
        ax1.plot(line_indices, solver_active, 'o-', color=self.colors['solver'], 
                linewidth=2, markersize=4, label='Your Solver', alpha=0.8)
        ax1.plot(line_indices, pf_active, 's-', color=self.colors['powerfactory'], 
                linewidth=2, markersize=4, label='PowerFactory', alpha=0.8)
        ax1.set_xlabel('Line Number')
        ax1.set_ylabel('Active Power (MW)')
        ax1.set_title('Active Power Flows', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Reactive Power Flows
        ax2.plot(line_indices, solver_reactive, 'o-', color=self.colors['solver'], 
                linewidth=2, markersize=4, label='Your Solver', alpha=0.8)
        ax2.plot(line_indices, pf_reactive, 's-', color=self.colors['powerfactory'], 
                linewidth=2, markersize=4, label='PowerFactory', alpha=0.8)
        ax2.set_xlabel('Line Number')
        ax2.set_ylabel('Reactive Power (MVAR)')
        ax2.set_title('Reactive Power Flows', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Active Power Errors
        active_error = np.abs(solver_active - pf_active)
        ax3.bar(line_indices, active_error, color=self.colors['error'], alpha=0.7)
        ax3.set_xlabel('Line Number')
        ax3.set_ylabel('Active Power Error (MW)')
        ax3.set_title('Active Power Flow Errors', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add error statistics
        max_p_error = np.max(active_error) if len(active_error) > 0 else 0
        mean_p_error = np.mean(active_error) if len(active_error) > 0 else 0
        ax3.text(0.02, 0.98, f'Max Error: {max_p_error:.2f} MW\nMean Error: {mean_p_error:.2f} MW', 
                transform=ax3.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Reactive Power Errors
        reactive_error = np.abs(solver_reactive - pf_reactive)
        ax4.bar(line_indices, reactive_error, color=self.colors['error'], alpha=0.7)
        ax4.set_xlabel('Line Number')
        ax4.set_ylabel('Reactive Power Error (MVAR)')
        ax4.set_title('Reactive Power Flow Errors', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add error statistics
        max_q_error = np.max(reactive_error) if len(reactive_error) > 0 else 0
        mean_q_error = np.mean(reactive_error) if len(reactive_error) > 0 else 0
        ax4.text(0.02, 0.98, f'Max Error: {max_q_error:.2f} MVAR\nMean Error: {mean_q_error:.2f} MVAR', 
                transform=ax4.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _create_generation_comparison(self, 
                                    solver_results: Dict[str, Any],
                                    pf_results: Dict[str, Any],
                                    scenario_info: Dict[str, Any]) -> plt.Figure:
        """Create generation comparison plot."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(f'Generation Comparison - {scenario_info.get("description", "Unknown Scenario")}', 
                    fontsize=16, fontweight='bold')
        
        # Extract data
        solver_gen = solver_results.get('generators', {})
        pf_gen = pf_results.get('generators', {})
        
        if not pf_gen:
            ax1.text(0.5, 0.5, 'No generation data available', ha='center', va='center')
            ax1.set_title('Generator Output')
            return fig
        
        # Get generator information
        n_gens = len(pf_gen.get('active_power', []))
        gen_indices = range(n_gens)
        gen_labels = [f"G{i+1}" for i in range(n_gens)]
        
        # PowerFactory generation
        pf_pg = np.array(pf_gen.get('active_power', []))
        pf_qg = np.array(pf_gen.get('reactive_power', []))
        
        # Solver generation (use defaults if not available)
        if solver_gen:
            solver_pg = np.array(solver_gen.get('active_power', np.zeros(n_gens)))[:n_gens]
            solver_qg = np.array(solver_gen.get('reactive_power', np.zeros(n_gens)))[:n_gens]
        else:
            solver_pg = np.zeros(n_gens)
            solver_qg = np.zeros(n_gens)
        
        # 1. Active Power Generation
        ax1.plot(gen_indices, solver_pg, 'o-', color=self.colors['solver'], 
                linewidth=2, markersize=6, label='Your Solver', alpha=0.8)
        ax1.plot(gen_indices, pf_pg, 's-', color=self.colors['powerfactory'], 
                linewidth=2, markersize=6, label='PowerFactory', alpha=0.8)
        ax1.set_xlabel('Generator Number')
        ax1.set_ylabel('Active Power (MW)')
        ax1.set_title('Active Power Generation', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Reactive Power Generation
        ax2.plot(gen_indices, solver_qg, 'o-', color=self.colors['solver'], 
                linewidth=2, markersize=6, label='Your Solver', alpha=0.8)
        ax2.plot(gen_indices, pf_qg, 's-', color=self.colors['powerfactory'], 
                linewidth=2, markersize=6, label='PowerFactory', alpha=0.8)
        ax2.set_xlabel('Generator Number')
        ax2.set_ylabel('Reactive Power (MVAR)')
        ax2.set_title('Reactive Power Generation', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Active Power Errors
        pg_error = np.abs(solver_pg - pf_pg)
        ax3.bar(gen_indices, pg_error, color=self.colors['error'], alpha=0.7)
        ax3.set_xlabel('Generator Number')
        ax3.set_ylabel('Active Power Error (MW)')
        ax3.set_title('Active Power Generation Errors', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add error statistics
        max_pg_error = np.max(pg_error) if len(pg_error) > 0 else 0
        mean_pg_error = np.mean(pg_error) if len(pg_error) > 0 else 0
        ax3.text(0.02, 0.98, f'Max Error: {max_pg_error:.2f} MW\nMean Error: {mean_pg_error:.2f} MW', 
                transform=ax3.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Reactive Power Errors
        qg_error = np.abs(solver_qg - pf_qg)
        ax4.bar(gen_indices, qg_error, color=self.colors['error'], alpha=0.7)
        ax4.set_xlabel('Generator Number')
        ax4.set_ylabel('Reactive Power Error (MVAR)')
        ax4.set_title('Reactive Power Generation Errors', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add error statistics
        max_qg_error = np.max(qg_error) if len(qg_error) > 0 else 0
        mean_qg_error = np.mean(qg_error) if len(qg_error) > 0 else 0
        ax4.text(0.02, 0.98, f'Max Error: {max_qg_error:.2f} MVAR\nMean Error: {mean_qg_error:.2f} MVAR', 
                transform=ax4.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _save_comparison_plots(self, figures: Dict[str, plt.Figure], scenario_info: Dict[str, Any]):
        """Save comparison plots to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_id = scenario_info.get('scenario_id', 'unknown')
        
        for plot_type, fig in figures.items():
            filename = f"comparison_{plot_type}_scenario_{scenario_id}_{timestamp}.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved {plot_type} comparison: {filepath}")
    
    def create_summary_report(self, 
                            comparison_results: List[Dict[str, Any]], 
                            save_path: Optional[str] = None) -> str:
        """Create a summary report of all comparisons.
        
        Args:
            comparison_results: List of comparison results from multiple scenarios
            save_path: Path to save the report (optional)
            
        Returns:
            The report content as a string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
PowerFactory vs Solver Comparison Report
Generated: {timestamp}

{'='*60}

SUMMARY STATISTICS:

Total Scenarios Analyzed: {len(comparison_results)}

"""
        
        # Add detailed statistics for each scenario
        for i, result in enumerate(comparison_results):
            scenario_info = result.get('scenario_info', {})
            errors = result.get('errors', {})
            
            report += f"""
Scenario {i+1}: {scenario_info.get('description', 'Unknown')}
{'-'*50}
Voltage Magnitude Errors:
  Max Error: {errors.get('voltage_mag_max', 0):.4f} pu
  Mean Error: {errors.get('voltage_mag_mean', 0):.4f} pu

Voltage Angle Errors:
  Max Error: {errors.get('voltage_ang_max', 0):.3f}°
  Mean Error: {errors.get('voltage_ang_mean', 0):.3f}°

Line Flow Errors:
  Active Power Max Error: {errors.get('active_power_max', 0):.2f} MW
  Reactive Power Max Error: {errors.get('reactive_power_max', 0):.2f} MVAR

Generation Errors:
  Active Power Max Error: {errors.get('gen_active_max', 0):.2f} MW
  Reactive Power Max Error: {errors.get('gen_reactive_max', 0):.2f} MVAR

"""
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📄 Comparison report saved: {save_path}")
        
        return report


def demo_comparison_visualization():
    """Demonstrate the PowerFactory comparison system."""
    print("\n🎨 PowerFactory Comparison Visualization Demo")
    print("=" * 50)
    
    # This would be integrated with your contingency analysis system
    print("This system creates detailed comparison plots between:")
    print("• Your three-phase load flow solver")
    print("• DIgSILENT PowerFactory results")
    print("\nPlot types generated:")
    print("1. 📊 Voltage Comparison (magnitudes, angles, errors)")
    print("2. ⚡ Line Flow Comparison (active, reactive power, errors)")
    print("3. 🏭 Generation Comparison (active, reactive power, errors)")
    print("\nPlots are automatically saved to 'Contingency Analysis/comparison_plots/'")


if __name__ == "__main__":
    demo_comparison_visualization()