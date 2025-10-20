# powerfactory_engine.py
"""
MODULE 1: PowerFactory Core Engine
==================================
Toutes les fonctions de connexion PowerFactory, gestion des éléments,
résolution de load flow, et fonctions utilitaires PowerFactory.
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# PowerFactory connection (using your exact path)
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10")
import powerfactory as pf

class PowerFactoryEngine:
    def __init__(self, project_name, study_case):
        self.project = project_name
        self.study = study_case
        self.app = None
        
    def connect_and_setup(self):
        """Connect to PowerFactory and setup project"""
        self.app = pf.GetApplication()
        if not self.app:
            raise Exception("❌ PowerFactory not running!")
        
        # Reset and connect (like First_model.py)
        if hasattr(self.app, "ResetCalculation"):
            self.app.ResetCalculation()
        
        if self.app.ActivateProject(self.project) != 0:
            raise Exception(f"❌ Project '{self.project}' not found!")
        
        # Find and activate study case
        study_case = None
        for case in self.app.GetProjectFolder("study").GetContents("*.IntCase"):
            if case.loc_name == self.study:
                study_case = case
                break
        
        if not study_case:
            raise Exception(f"❌ Study case '{self.study}' not found!")
        
        study_case.Activate()
        print(f"✅ Connected: {self.project} | {self.study}")
        return self.app
    
    def get_all_system_elements(self):
        """Get all generators, loads, lines, transformers"""
        if not self.app:
            raise Exception("❌ PowerFactory not connected!")
            
        elements = {
            'generators': self.app.GetCalcRelevantObjects("*.ElmSym"),
            'loads': self.app.GetCalcRelevantObjects("*.ElmLod"),
            'lines': self.app.GetCalcRelevantObjects("*.ElmLne"),
            'transformers_2w': self.app.GetCalcRelevantObjects("*.ElmTr2"),
            'transformers_3w': self.app.GetCalcRelevantObjects("*.ElmTr3"),
            'buses': self.app.GetCalcRelevantObjects("*.ElmTerm")
        }
        
        return elements
    
    def solve_power_flow(self, tolerance=0.1, max_iter=50):
        """Solve power flow with robust settings"""
        try:
            comLdf = self.app.GetFromStudyCase("ComLdf")
            comLdf.iopt_net = 0      # AC load flow
            comLdf.iopt_at = 1       # Automatic tap adjustment
            comLdf.errlf = tolerance # Tolerance
            comLdf.maxiter = max_iter # Maximum iterations
            
            ierr = comLdf.Execute()
            return ierr == 0
        except:
            return False
    
    def create_outage(self, element_name, element_type):
        """Create outage for specific element"""
        if element_type == "generator":
            elements = self.app.GetCalcRelevantObjects("*.ElmSym")
        elif element_type == "line":
            elements = self.app.GetCalcRelevantObjects("*.ElmLne")
        elif element_type == "load":
            elements = self.app.GetCalcRelevantObjects("*.ElmLod")
        elif element_type == "transformer":
            elements = self.app.GetCalcRelevantObjects("*.ElmTr2")
            elements.extend(self.app.GetCalcRelevantObjects("*.ElmTr3"))
        else:
            return None, None
        
        # Find the element
        target_element = None
        for elem in elements:
            if element_name.lower() in safe_get_name(elem).lower():
                target_element = elem
                break
        
        if not target_element:
            return None, None
        
        # Store original state and create outage
        original_outserv = getattr(target_element, 'outserv', 0)
        target_element.SetAttribute('outserv', 1)
        
        return target_element, original_outserv
    
    def restore_element(self, element_obj, original_state):
        """Restore element from outage"""
        try:
            element_obj.SetAttribute('outserv', original_state)
            return True
        except:
            return False

# ── Helper Functions (from your working scripts) ───────────────────────────
def has(o, a):
    """Safely check if object has attribute"""
    try:
        return o.HasAttribute(a) if o else False
    except:
        return False

def get(o, a, d=np.nan):
    """Safely get attribute value"""
    try:
        return o.GetAttribute(a) if has(o, a) else d
    except:
        return d

def safe_get_name(obj):
    """Safely get object name"""
    try:
        return obj.loc_name if obj else "Unknown"
    except:
        return "Unknown"

def safe_get_class(obj):
    """Safely get object class name"""
    try:
        return obj.GetClassName() if obj else "Unknown"
    except:
        return "Unknown"