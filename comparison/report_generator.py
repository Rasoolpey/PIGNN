"""
Report generation utilities for comparison results.

This module provides tools to generate comprehensive reports from 
DIgSILENT vs solver comparison results, including:
- HTML reports with plots
- LaTeX tables
- Summary statistics
- Visualization of error distributions
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.error_metrics import ComparisonResults


class ComparisonReportGenerator:
    """
    Generate comprehensive reports from comparison results.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "Contingency Analysis/reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for optional dependencies
        self.has_matplotlib = self._check_matplotlib()
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available for plotting."""
        try:
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            return False
    
    def generate_scenario_report(self, 
                               scenario_name: str,
                               comparison: ComparisonResults,
                               include_plots: bool = True) -> str:
        """
        Generate detailed report for a single scenario comparison.
        
        Args:
            scenario_name: Name of the scenario
            comparison: ComparisonResults object
            include_plots: Whether to include error distribution plots
            
        Returns:
            Path to generated HTML report
        """
        # Generate HTML report
        html_content = self._generate_html_report(scenario_name, comparison, include_plots)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"scenario_report_{scenario_name}_{timestamp}.html"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_file)
    
    def generate_batch_report(self,
                            comparisons: Dict[str, ComparisonResults],
                            summary_stats: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate summary report for batch comparisons.
        
        Args:
            comparisons: Dictionary of scenario comparisons
            summary_stats: Optional summary statistics
            
        Returns:
            Path to generated HTML report
        """
        html_content = self._generate_batch_html_report(comparisons, summary_stats)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"batch_report_{timestamp}.html"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_file)
    
    def _generate_html_report(self, 
                            scenario_name: str, 
                            comparison: ComparisonResults,
                            include_plots: bool = True) -> str:
        """Generate HTML content for single scenario report."""
        
        # Get quality assessment
        from comparison.error_metrics import ErrorCalculator
        error_calc = ErrorCalculator()
        quality = error_calc.assess_convergence_quality(comparison)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Scenario Comparison Report: {scenario_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; }}
                .error {{ color: #d32f2f; }}
                .warning {{ color: #f57c00; }}
                .good {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Power Flow Comparison Report</h1>
                <h2>Scenario: {scenario_name}</h2>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h3>Overall Quality Assessment</h3>
                <div class="metric">
                    <strong>Overall Quality:</strong> 
                    <span class="{self._get_quality_class(quality['overall_quality'])}">
                        {quality['overall_quality'].upper()}
                    </span>
                </div>
                <div class="metric">
                    <strong>Voltage Quality:</strong> 
                    <span class="{self._get_quality_class(quality['voltage_quality'])}">
                        {quality['voltage_quality'].upper()}
                    </span>
                </div>
                <div class="metric">
                    <strong>Flow Quality:</strong> 
                    <span class="{self._get_quality_class(quality['flow_quality'])}">
                        {quality['flow_quality'].upper()}
                    </span>
                </div>
                {self._format_issues_list(quality['issues'])}
            </div>
            
            <div class="section">
                <h3>Convergence Information</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Solver Converged</td><td>{comparison.solver_converged}</td></tr>
                    <tr><td>Iterations</td><td>{comparison.solver_iterations}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h3>Voltage Error Summary</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Unit</th></tr>
                    <tr><td>Maximum Magnitude Error</td><td>{comparison.max_voltage_error_pu:.6f}</td><td>pu</td></tr>
                    <tr><td>Maximum Percentage Error</td><td>{comparison.max_voltage_error_percent:.4f}</td><td>%</td></tr>
                    <tr><td>RMS Magnitude Error</td><td>{comparison.rms_voltage_error_pu:.6f}</td><td>pu</td></tr>
                    <tr><td>Maximum Angle Error</td><td>{comparison.max_angle_error_deg:.4f}</td><td>degrees</td></tr>
                    <tr><td>RMS Angle Error</td><td>{comparison.rms_angle_error_deg:.4f}</td><td>degrees</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h3>Power Flow Error Summary</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Unit</th></tr>
                    <tr><td>Maximum P Flow Error</td><td>{comparison.max_p_flow_error_percent:.4f}</td><td>%</td></tr>
                    <tr><td>Maximum Q Flow Error</td><td>{comparison.max_q_flow_error_percent:.4f}</td><td>%</td></tr>
                    <tr><td>RMS P Flow Error</td><td>{comparison.rms_p_flow_error_percent:.4f}</td><td>%</td></tr>
                    <tr><td>RMS Q Flow Error</td><td>{comparison.rms_q_flow_error_percent:.4f}</td><td>%</td></tr>
                </table>
            </div>
        """
        
        # Add plots if requested and matplotlib is available
        if include_plots and self.has_matplotlib and comparison.voltage_mag_error_pu is not None:
            plot_paths = self._generate_error_plots(scenario_name, comparison)
            html += self._format_plots_section(plot_paths)
        
        # Add detailed error tables if data is available
        if comparison.voltage_mag_error_pu is not None and comparison.bus_names:
            html += self._format_voltage_error_table(comparison)
        
        if comparison.p_flow_error_mw is not None and comparison.branch_names:
            html += self._format_flow_error_table(comparison)
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_batch_html_report(self,
                                  comparisons: Dict[str, ComparisonResults],
                                  summary_stats: Optional[Dict[str, Any]] = None) -> str:
        """Generate HTML content for batch report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; }}
                .error {{ color: #d32f2f; }}
                .warning {{ color: #f57c00; }}
                .good {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .scenario-row:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Batch Power Flow Comparison Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Total Scenarios: {len(comparisons)}</p>
            </div>
        """
        
        # Add summary statistics if available
        if summary_stats:
            html += self._format_summary_stats_section(summary_stats)
        
        # Add scenario comparison table
        html += self._format_scenario_comparison_table(comparisons)
        
        # Add worst performers section
        html += self._format_worst_performers_section(comparisons)
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_error_plots(self, scenario_name: str, comparison: ComparisonResults) -> Dict[str, str]:
        """Generate error distribution plots."""
        if not self.has_matplotlib:
            return {}
        
        import matplotlib.pyplot as plt
        
        plot_paths = {}
        
        # Voltage magnitude error histogram
        if comparison.voltage_mag_error_pu is not None:
            plt.figure(figsize=(10, 6))
            plt.hist(comparison.voltage_mag_error_pu, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Voltage Magnitude Error (pu)')
            plt.ylabel('Frequency')
            plt.title(f'Voltage Magnitude Error Distribution - {scenario_name}')
            plt.grid(True, alpha=0.3)
            
            plot_file = self.output_dir / f"voltage_error_hist_{scenario_name}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['voltage_hist'] = str(plot_file.name)
        
        # Voltage angle error histogram
        if comparison.voltage_angle_error_deg is not None:
            plt.figure(figsize=(10, 6))
            plt.hist(comparison.voltage_angle_error_deg, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Voltage Angle Error (degrees)')
            plt.ylabel('Frequency')
            plt.title(f'Voltage Angle Error Distribution - {scenario_name}')
            plt.grid(True, alpha=0.3)
            
            plot_file = self.output_dir / f"angle_error_hist_{scenario_name}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['angle_hist'] = str(plot_file.name)
        
        # Power flow error scatter plot
        if (comparison.p_flow_error_percent is not None and 
            comparison.q_flow_error_percent is not None):
            plt.figure(figsize=(10, 8))
            plt.scatter(comparison.p_flow_error_percent, comparison.q_flow_error_percent, 
                       alpha=0.6)
            plt.xlabel('P Flow Error (%)')
            plt.ylabel('Q Flow Error (%)')
            plt.title(f'Power Flow Error Scatter - {scenario_name}')
            plt.grid(True, alpha=0.3)
            
            plot_file = self.output_dir / f"flow_error_scatter_{scenario_name}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['flow_scatter'] = str(plot_file.name)
        
        return plot_paths
    
    def _get_quality_class(self, quality: str) -> str:
        """Get CSS class for quality level."""
        if quality in ['excellent', 'good']:
            return 'good'
        elif quality == 'acceptable':
            return 'warning'
        else:
            return 'error'
    
    def _format_issues_list(self, issues: List[str]) -> str:
        """Format issues list as HTML."""
        if not issues:
            return '<div class="metric good">No issues detected</div>'
        
        html = '<div class="metric error"><strong>Issues:</strong><ul>'
        for issue in issues:
            html += f'<li>{issue}</li>'
        html += '</ul></div>'
        return html
    
    def _format_plots_section(self, plot_paths: Dict[str, str]) -> str:
        """Format plots section of HTML report."""
        if not plot_paths:
            return ""
        
        html = '<div class="section"><h3>Error Distribution Plots</h3>'
        
        for plot_type, plot_file in plot_paths.items():
            html += f'<div><img src="{plot_file}" alt="{plot_type}" style="max-width: 100%; margin: 10px 0;"></div>'
        
        html += '</div>'
        return html
    
    def _format_voltage_error_table(self, comparison: ComparisonResults) -> str:
        """Format detailed voltage error table."""
        html = '<div class="section"><h3>Detailed Voltage Errors by Bus</h3><table>'
        html += '<tr><th>Bus Name</th><th>Mag Error (pu)</th><th>Mag Error (%)</th><th>Angle Error (deg)</th></tr>'
        
        for i, bus in enumerate(comparison.bus_names[:20]):  # Limit to first 20 buses
            mag_err = comparison.voltage_mag_error_pu[i]
            mag_err_pct = comparison.voltage_mag_error_percent[i]
            ang_err = comparison.voltage_angle_error_deg[i]
            
            html += f'<tr><td>{bus}</td><td>{mag_err:.6f}</td><td>{mag_err_pct:.4f}</td><td>{ang_err:.4f}</td></tr>'
        
        if len(comparison.bus_names) > 20:
            html += f'<tr><td colspan="4">... and {len(comparison.bus_names) - 20} more buses</td></tr>'
        
        html += '</table></div>'
        return html
    
    def _format_flow_error_table(self, comparison: ComparisonResults) -> str:
        """Format detailed flow error table."""
        html = '<div class="section"><h3>Detailed Flow Errors by Branch</h3><table>'
        html += '<tr><th>Branch</th><th>P Error (MW)</th><th>Q Error (MVAr)</th><th>P Error (%)</th><th>Q Error (%)</th></tr>'
        
        for i, branch in enumerate(comparison.branch_names[:20]):  # Limit to first 20 branches
            p_err_mw = comparison.p_flow_error_mw[i]
            q_err_mvar = comparison.q_flow_error_mvar[i]
            p_err_pct = comparison.p_flow_error_percent[i]
            q_err_pct = comparison.q_flow_error_percent[i]
            
            html += f'<tr><td>{branch}</td><td>{p_err_mw:.4f}</td><td>{q_err_mvar:.4f}</td>'
            html += f'<td>{p_err_pct:.4f}</td><td>{q_err_pct:.4f}</td></tr>'
        
        if len(comparison.branch_names) > 20:
            html += f'<tr><td colspan="5">... and {len(comparison.branch_names) - 20} more branches</td></tr>'
        
        html += '</table></div>'
        return html
    
    def _format_summary_stats_section(self, summary_stats: Dict[str, Any]) -> str:
        """Format summary statistics section."""
        html = '<div class="section"><h3>Summary Statistics</h3>'
        
        html += f"""
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Scenarios</td><td>{summary_stats.get('total_scenarios', 0)}</td></tr>
            <tr><td>Converged Scenarios</td><td>{summary_stats.get('converged_scenarios', 0)}</td></tr>
            <tr><td>Convergence Rate</td><td>{summary_stats.get('convergence_rate', 0):.2%}</td></tr>
            <tr><td>Average Iterations</td><td>{summary_stats.get('avg_iterations', 0):.1f}</td></tr>
            <tr><td>Max Iterations</td><td>{summary_stats.get('max_iterations', 0)}</td></tr>
        </table>
        """
        
        # Add error statistics
        if 'voltage_error_stats' in summary_stats:
            v_stats = summary_stats['voltage_error_stats']
            html += f"""
            <h4>Voltage Error Statistics</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Max Absolute Error (pu)</td><td>{v_stats.get('max_abs', 0):.6f}</td></tr>
                <tr><td>RMS Error (pu)</td><td>{v_stats.get('rms', 0):.6f}</td></tr>
                <tr><td>Mean Error (pu)</td><td>{v_stats.get('mean', 0):.6f}</td></tr>
                <tr><td>Std Deviation (pu)</td><td>{v_stats.get('std', 0):.6f}</td></tr>
            </table>
            """
        
        html += '</div>'
        return html
    
    def _format_scenario_comparison_table(self, comparisons: Dict[str, ComparisonResults]) -> str:
        """Format scenario comparison table."""
        html = '<div class="section"><h3>Scenario Comparison Summary</h3><table>'
        html += """
        <tr>
            <th>Scenario</th>
            <th>Converged</th>
            <th>Iterations</th>
            <th>Max V Error (pu)</th>
            <th>Max Angle Error (deg)</th>
            <th>Max P Flow Error (%)</th>
            <th>Max Q Flow Error (%)</th>
        </tr>
        """
        
        for name, comp in comparisons.items():
            converged_icon = "✓" if comp.solver_converged else "✗"
            converged_class = "good" if comp.solver_converged else "error"
            
            html += f"""
            <tr class="scenario-row">
                <td>{name}</td>
                <td class="{converged_class}">{converged_icon}</td>
                <td>{comp.solver_iterations}</td>
                <td>{comp.max_voltage_error_pu:.6f}</td>
                <td>{comp.max_angle_error_deg:.4f}</td>
                <td>{comp.max_p_flow_error_percent:.4f}</td>
                <td>{comp.max_q_flow_error_percent:.4f}</td>
            </tr>
            """
        
        html += '</table></div>'
        return html
    
    def _format_worst_performers_section(self, comparisons: Dict[str, ComparisonResults]) -> str:
        """Format worst performers section."""
        # Find worst scenarios by different metrics
        scenarios_by_voltage_error = sorted(
            [(name, comp.max_voltage_error_pu) for name, comp in comparisons.items()],
            key=lambda x: x[1], reverse=True)[:5]
        
        scenarios_by_flow_error = sorted(
            [(name, comp.max_p_flow_error_percent) for name, comp in comparisons.items()],
            key=lambda x: x[1], reverse=True)[:5]
        
        html = '<div class="section"><h3>Worst Performing Scenarios</h3>'
        
        html += '<h4>Highest Voltage Errors</h4><table>'
        html += '<tr><th>Rank</th><th>Scenario</th><th>Max Voltage Error (pu)</th></tr>'
        for i, (name, error) in enumerate(scenarios_by_voltage_error, 1):
            html += f'<tr><td>{i}</td><td>{name}</td><td>{error:.6f}</td></tr>'
        html += '</table>'
        
        html += '<h4>Highest Flow Errors</h4><table>'
        html += '<tr><th>Rank</th><th>Scenario</th><th>Max P Flow Error (%)</th></tr>'
        for i, (name, error) in enumerate(scenarios_by_flow_error, 1):
            html += f'<tr><td>{i}</td><td>{name}</td><td>{error:.4f}</td></tr>'
        html += '</table>'
        
        html += '</div>'
        return html