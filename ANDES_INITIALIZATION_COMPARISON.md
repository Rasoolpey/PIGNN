# ANDES vs Current Implementation: Generator Initialization Comparison

## Key Findings from ANDES GENROU Code

### 1. **Initialization Strategy** (Lines 88-155 in genrou.py)

ANDES uses a **multi-step algebraic approach** to compute initial states:

```python
# Step 1: Complex phasor calculations
_V = v * exp(1j * a)                    # Bus voltage phasor
_S = p0 - 1j * q0                       # Complex power
_It = _S / conj(_V)                     # Terminal current
_Zs = ra + 1j * xd2                     # Equivalent impedance (uses xd")
_Is = _It + _V / _Zs                    # Equivalent current source

# Step 2: Subtransient flux linkage
psi20 = _Is * _Zs                       # ψ" in stator frame
psi20_abs = abs(psi20)
psi20_arg = arg(psi20)

# Step 3: Saturation calculation
Se0 = Indicator(psi20_abs >= SAT_A) * (psi20_abs - SAT_A)^2 * SAT_B / psi20_abs

# Step 4: Rotor angle from geometry
_a = psi20_abs * (1 + Se0*gqd)
_b = abs(_It) * (xq2 - xq)
delta0 = atan(_b*cos(θ) / (_b*sin(θ) - _a)) + psi20_arg

# Step 5: Park transformation to dq frame
_Tdq = cos(delta0) - 1j*sin(delta0)
psi20_dq = psi20 * _Tdq
It_dq = conj(_It * _Tdq)

psi2d0 = re(psi20_dq)    # d-axis subtransient flux
psi2q0 = -im(psi20_dq)   # q-axis subtransient flux
Id0 = im(It_dq)          # d-axis current
Iq0 = re(It_dq)          # q-axis current

# Step 6: Terminal voltage in dq frame
vd0 = psi2q0 + xq2*Iq0 - ra*Id0
vq0 = psi2d0 - xd2*Id0 - ra*Iq0

# Step 7: Initial field voltage and torque
vf0 = (Se0 + 1)*psi2d0 + (xd - xd2)*Id0
tm0 = (vq0 + ra*Iq0)*Iq0 + (vd0 + ra*Id0)*Id0

# Step 8: Transient voltages
e1q0 = Id0*(-xd + xd1) - Se0*psi2d0 + vf0    # Eq' initial
e1d0 = Iq0*(xq - xq1) - Se0*gqd*psi2q0       # Ed' initial

# Step 9: Subtransient voltages  
e2d0 = Id0*(xl - xd) - Se0*psi2d0 + vf0      # Ed" initial
e2q0 = -Iq0*(xl - xq) - Se0*gqd*psi2q0       # Eq" initial
```

### 2. **Critical Differences from Our Implementation**

#### **A. Impedance Used for Initialization**
- **ANDES**: Uses **xd"** (subtransient reactance) for equivalent circuit
  ```python
  _Zs = ra + 1j * xd2  # xd2 = xd"
  ```
- **OUR CODE**: Uses **xd'** (transient reactance)
  ```python
  Z = self.params.ra_pu + 1j * self.params.xd_prime_pu
  ```
- **Impact**: Different impedance → different rotor angle → different dq currents → wrong flux linkages

#### **B. Saturation Handling**
- **ANDES**: Computes saturation **Se0** based on subtransient flux magnitude
  ```python
  Se0 = (psi20_abs - SAT_A)^2 * SAT_B / psi20_abs  # if psi20_abs >= SAT_A
  ```
  Then uses Se0 in ALL flux/voltage calculations
- **OUR CODE**: Doesn't account for saturation in initialization
- **Impact**: Wrong flux linkages, especially for heavily loaded generators (G 05 at 169%)

#### **C. Ed' and Eq' Calculation**
- **ANDES**: Derives from subtransient values with saturation correction
  ```python
  e1q0 = Id0*(-xd + xd1) - Se0*psi2d0 + vf0
  e1d0 = Iq0*(xq - xq1) - Se0*gqd*psi2q0
  ```
- **OUR CODE**: Simplified calculation without saturation
  ```python
  Eq_prime = abs(E_internal_dq_prime)
  Ed_prime = 0.0  # Assumed zero (wrong!)
  ```
- **Impact**: Ed' initialized incorrectly → large dEd'/dt residuals

#### **D. Flux Linkage Equations**
- **ANDES**: Uses algebraic constraints (Flux0 class)
  ```python
  psid = ra*Iq + vq   # Steady-state algebraic
  psiq = -(ra*Id + vd)
  ```
  These are **algebraic variables**, not differential states!
- **OUR CODE**: May be treating flux as differential states
- **Impact**: Extra equations, potential initialization mismatch

### 3. **State Variables in ANDES**

#### **Differential States** (time derivatives)
```python
delta: d(delta)/dt = 2*pi*fn*(omega - 1)
omega: M*d(omega)/dt = tm - te - D*(omega - 1)
e1q:   Td10*d(e1q)/dt = -XadIfd + vf     # Eq' transient
e1d:   Tq10*d(e1d)/dt = -XaqI1q           # Ed' transient
e2d:   Td20*d(e2d)/dt = -e2d + e1q - (xd1-xl)*Id   # Ed" subtransient
e2q:   Tq20*d(e2q)/dt = -e2q + e1d + (xq1-xl)*Iq   # Eq" subtransient
```

#### **Algebraic Variables** (constraints)
```python
vd:  v*sin(delta - a) - vd = 0
vq:  v*cos(delta - a) - vq = 0
Id:  vd + ra*Id - xq2*Iq - psi2d = 0  # (completed by Flux0)
Iq:  vq - ra*Iq - xd2*Id + psi2q = 0
psid: ra*Iq + vq - psid = 0           # Algebraic (Flux0)
psiq: ra*Id + vd + psiq = 0
psi2d: gd1*e1q + gd2*(xd1-xl)*e2d - psi2d = 0
psi2q: gq1*e1d + (1-gq1)*e2q - psi2q = 0
te:   psid*Iq - psiq*Id - te = 0
XadIfd: e1q + (xd-xd1)*(gd1*Id - gd2*e2d + gd2*e1q) + Se*psi2d - XadIfd = 0
XaqI1q: e1d + (xq-xq1)*(gq2*e1d - gq2*e2q - gq1*Iq) + Se*psi2q*gqd - XaqI1q = 0
```

### 4. **Why Our Ed' Residuals are Large**

Looking at ANDES initialization for Ed':
```python
# ANDES computes Ed' initial value as:
e1d0 = Iq0*(xq - xq1) - Se0*gqd*psi2q0

# Where:
# - Iq0: q-axis current (computed from complex phasors with xd")
# - Se0: saturation at operating point
# - gqd = (xq - xl)/(xd - xl)
# - psi2q0: q-axis subtransient flux
```

Our code likely:
1. Uses **wrong Iq** (because we use xd' not xd" for rotor angle)
2. **Ignores saturation** (Se0 = 0)
3. **Sets Ed' = 0** (round rotor assumption, but G 05 is salient pole hydro!)

For **G 05** (hydro, heavily loaded):
- Has **xq ≠ xd** (salient pole)
- Operates at **169% power** (high saturation)
- Our Ed' = 0 → ANDES Ed' = significant non-zero value
- Result: **dEd'/dt = (Ed'_target - 0) / Tq0' = large residual**

### 5. **Recommended Fixes**

#### **Option 1: Match ANDES Exactly** (Most Accurate)
```python
# In generator_models.py initialize():

# 1. Use xd" for equivalent circuit
Z = self.params.ra_pu + 1j * self.params.xd_double_pu  # NOT xd_prime!

# 2. Compute psi" and saturation
psi20 = I_s * Z
psi20_abs = abs(psi20)
Se0 = compute_saturation(psi20_abs, S10, S12)

# 3. Solve for rotor angle with geometry
a = psi20_abs * (1 + Se0*gqd)
b = abs(I_t) * (xq" - xq)
delta = atan(b*cos(theta) / (b*sin(theta) - a)) + angle(psi20)

# 4. Transform to dq frame
psi2d0 = ...
psi2q0 = ...
Id0 = ...
Iq0 = ...

# 5. Compute transient voltages with saturation
e1q0 = Id0*(-xd + xd1) - Se0*psi2d0 + vf0
e1d0 = Iq0*(xq - xq1) - Se0*gqd*psi2q0

# 6. Initialize states
self.states[2] = e1q0  # Eq'
self.states[3] = e1d0  # Ed' (NOT ZERO!)
```

#### **Option 2: Use ANDES Directly** (Simplest)
- Import ANDES GENROU initialization routines
- Call their v_str (initial value) functions
- Use their algebraic constraint formulation

#### **Option 3: Simplified Fix** (Quick)
```python
# Just fix Ed' calculation
if Tq0_prime > 0:  # If has q-axis transient
    # Compute Ed' from steady-state equation
    Ed_prime = Iq * (xq - xq1)  # Basic formula
    self.states[3] = Ed_prime
else:
    self.states[3] = 0.0  # Round rotor
```

### 6. **Action Items**

1. ✅ **Understand ANDES approach** (DONE - reviewed genrou.py)
2. ⏳ **Implement xd" initialization** (change from xd' to xd")
3. ⏳ **Add saturation calculation** (Se0 based on flux magnitude)
4. ⏳ **Fix Ed' initialization** (use ANDES formula with saturation)
5. ⏳ **Test with G 05** (verify dEd'/dt ≈ 0)
6. ⏳ **Run full RMS simulation** (Newton should converge)

### 7. **Expected Impact**

After implementing ANDES initialization:
- **Current**: max |f| = 4.7 pu/s (G 05 Ed')
- **Expected**: max |f| < 0.1 pu/s (all states)
- **Newton convergence**: 3-5 iterations (not diverging!)
- **Simulation success**: Full 1s run without failures

---

## References

- ANDES GENROU: `explainations/andes-master/andes/models/synchronous/genrou.py`
- OpenIPSL Reference: https://github.com/OpenIPSL/OpenIPSL/blob/master/OpenIPSL/Electrical/Machines/PSSE/GENROU.mo
- Our implementation: `RMS_Analysis/generator_models.py` (lines 80-230)
