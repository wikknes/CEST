import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from scipy.fft import fft, fftfreq, ifft
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

@dataclass
class ResourceVariable:
    """Represents a single resource variable (protein, water, temperature, shelter)"""
    name: str
    required: np.ndarray  # Time series of required amounts
    available: np.ndarray  # Time series of available amounts
    critical_threshold: float  # Critical mismatch threshold
    weight: float = 1.0  # Importance weight in cross-entropy calculation

class CrossEntropySurvivalModel:
    """
    Complete implementation of Cross-Entropy Survival Theory with Laplace and Fourier analysis
    """
    
    def __init__(self, 
                 adaptation_rate: float = 0.3,  # k parameter (1/days)
                 sensitivity: float = 0.5,      # alpha parameter
                 critical_threshold: float = 10.0,  # D_critical
                 dt: float = 0.1):              # Time step (days)
        
        self.k = adaptation_rate
        self.alpha = sensitivity
        self.D_critical = critical_threshold
        self.dt = dt
        
        # Storage for analysis results
        self.time_series = {}
        self.fourier_results = {}
        self.laplace_results = {}
        
    def generate_cat_resource_data(self, days: int = 365) -> Dict[str, ResourceVariable]:
        """
        Generate realistic time series data for cat resource requirements and availability
        """
        t = np.arange(0, days, self.dt)
        n_points = len(t)
        
        # Add random seed for reproducibility
        np.random.seed(42)
        
        # Protein resource (5g/kg body weight/day baseline)
        protein_required = 5.0 * np.ones(n_points)  # Constant requirement
        protein_available = (4.0 + 
                           2.0 * np.sin(2*np.pi*t/365) +  # Seasonal variation
                           0.5 * np.sin(2*np.pi*t/1) +    # Daily hunting cycles
                           0.3 * np.random.normal(0, 1, n_points))  # Noise
        
        # Water resource (50ml/kg body weight/day baseline)
        water_required = 50.0 * np.ones(n_points)
        water_available = (45.0 + 
                         10.0 * np.sin(2*np.pi*t/365 + np.pi) +  # Opposite seasonal pattern
                         5.0 * np.sin(2*np.pi*t/7) +  # Weekly patterns
                         2.0 * np.random.normal(0, 1, n_points))
        
        # Temperature resource (optimal range 18-22°C)
        temp_required = 20.0 * np.ones(n_points)  # Optimal temperature
        temp_available = (20.0 + 
                        8.0 * np.sin(2*np.pi*t/365 - np.pi/2) +  # Seasonal temperature
                        3.0 * np.sin(2*np.pi*t/1) +  # Daily temperature cycle
                        1.0 * np.random.normal(0, 1, n_points))
        
        # Shelter resource (1m² per cat minimum)
        shelter_required = 1.0 * np.ones(n_points)
        shelter_available = (1.2 + 
                           0.3 * np.sin(2*np.pi*t/365) +  # Seasonal shelter availability
                           0.1 * np.sin(2*np.pi*t/30) +   # Monthly urban development cycles
                           0.2 * np.random.normal(0, 1, n_points))
        
        # Ensure non-negative values
        protein_available = np.maximum(protein_available, 0.1)
        water_available = np.maximum(water_available, 1.0)
        shelter_available = np.maximum(shelter_available, 0.1)
        
        resources = {
            'protein': ResourceVariable('Protein (g/kg/day)', protein_required, protein_available, 3.0, 1.0),
            'water': ResourceVariable('Water (ml/kg/day)', water_required, water_available, 20.0, 0.8),
            'temperature': ResourceVariable('Temperature (°C)', temp_required, temp_available, 5.0, 0.6),
            'shelter': ResourceVariable('Shelter (m²)', shelter_required, shelter_available, 0.5, 0.7)
        }
        
        self.time_vector = t
        self.resources = resources
        return resources
    
    def calculate_cross_entropy_components(self) -> Dict[str, np.ndarray]:
        """
        Calculate cross-entropy for each resource variable
        H_j(t) = -p_j(t) * log(q_j(t))
        """
        cross_entropy_components = {}
        
        for name, resource in self.resources.items():
            # Normalize to probability distributions
            p_j = resource.required / np.sum(resource.required)
            q_j = resource.available / np.sum(resource.available)
            
            # Avoid log(0) by adding small epsilon
            epsilon = 1e-10
            q_j = np.maximum(q_j, epsilon)
            
            # Calculate cross-entropy component
            H_j = -p_j * np.log(q_j)
            cross_entropy_components[name] = H_j * resource.weight
            
        # Total cross-entropy
        total_H = sum(cross_entropy_components.values())
        cross_entropy_components['total'] = total_H
        
        self.cross_entropy_components = cross_entropy_components
        return cross_entropy_components
    
    def fourier_analysis(self) -> Dict[str, Dict]:
        """
        Perform Fourier analysis on each resource variable and cross-entropy components
        """
        fourier_results = {}
        
        for name, H_component in self.cross_entropy_components.items():
            # Compute FFT
            H_fft = fft(H_component)
            freqs = fftfreq(len(H_component), d=self.dt)
            
            # Power spectral density
            psd = np.abs(H_fft)**2
            
            # Find dominant frequencies (excluding DC component)
            dominant_indices = np.argsort(psd[1:])[::-1][:5] + 1
            dominant_freqs = freqs[dominant_indices]
            dominant_periods = 1 / np.abs(dominant_freqs)  # Convert to periods in days
            dominant_powers = psd[dominant_indices]
            
            fourier_results[name] = {
                'frequencies': freqs,
                'spectrum': H_fft,
                'power': psd,
                'dominant_frequencies': dominant_freqs,
                'dominant_periods': dominant_periods,
                'dominant_powers': dominant_powers,
                'total_power': np.sum(psd)
            }
        
        self.fourier_results = fourier_results
        return fourier_results
    
    def laplace_analysis_scenarios(self) -> Dict[str, Dict]:
        """
        Analyze different environmental disturbance scenarios using Laplace transforms
        """
        scenarios = {}
        
        # Scenario A: Urban food source removal (step function)
        def step_response(t, D0=1.0, delta_D=2.0):
            """Analytical Laplace solution for step disturbance"""
            steady_state = delta_D / self.k
            transient = (D0 - steady_state) * np.exp(-self.k * t)
            return steady_state + transient
        
        # Scenario B: Climate change (ramp function)
        def ramp_response(t, D0=1.0, alpha_ramp=0.1):
            """Analytical Laplace solution for ramp disturbance"""
            steady_slope = alpha_ramp / self.k
            steady_offset = alpha_ramp / (self.k**2)
            transient = (D0 + steady_offset) * np.exp(-self.k * t)
            return steady_slope * t - steady_offset + transient
        
        # Scenario C: Habitat fragmentation (impulse response)
        def impulse_response(t, D0=1.0, impulse_magnitude=5.0):
            """Analytical Laplace solution for impulse disturbance"""
            return (D0 + impulse_magnitude) * np.exp(-self.k * t)
        
        # Generate time vectors for analysis
        t_analysis = np.linspace(0, 50, 500)  # 50 days, high resolution
        
        scenarios['step_disturbance'] = {
            'time': t_analysis,
            'response': step_response(t_analysis),
            'description': 'Urban food source removal',
            'transfer_function': f"G(s) = 1/(s + {self.k})",
            'final_value': 2.0 / self.k
        }
        
        scenarios['ramp_disturbance'] = {
            'time': t_analysis,
            'response': ramp_response(t_analysis),
            'description': 'Climate change (gradual warming)',
            'transfer_function': f"G(s) = 1/(s²(s + {self.k}))",
            'final_value': 'Unbounded (extinction inevitable)'
        }
        
        scenarios['impulse_disturbance'] = {
            'time': t_analysis,
            'response': impulse_response(t_analysis),
            'description': 'Habitat fragmentation shock',
            'transfer_function': f"G(s) = 1/(s + {self.k})",
            'final_value': 0  # Returns to baseline
        }
        
        self.laplace_scenarios = scenarios
        return scenarios
    
    def population_dynamics_simulation(self, scenario_name: str = 'step_disturbance') -> Dict:
        """
        Simulate population dynamics based on cross-entropy deficit using Laplace analysis
        """
        scenario = self.laplace_scenarios[scenario_name]
        D_trajectory = scenario['response']
        t = scenario['time']
        
        # Population parameters
        P0 = 100  # Initial population
        r = 0.1   # Intrinsic growth rate (1/day)
        K = 1000  # Carrying capacity
        
        def population_ode(P, t_current, D_interp_func):
            """ODE for population dynamics"""
            D_current = D_interp_func(t_current)
            
            # Logistic growth with cross-entropy mortality
            growth_term = r * P * (1 - P/K)
            mortality_term = self.alpha * P * max(0, D_current - self.D_critical)
            
            dPdt = growth_term - mortality_term
            return dPdt
        
        # Create interpolation function for D(t)
        from scipy.interpolate import interp1d
        D_interp = interp1d(t, D_trajectory, kind='linear', fill_value='extrapolate')
        
        # Solve ODE
        P_trajectory = integrate.odeint(population_ode, P0, t, args=(D_interp,))
        P_trajectory = P_trajectory.flatten()
        
        # Calculate extinction probability
        extinction_prob = np.exp(-self.alpha * np.maximum(0, D_trajectory - self.D_critical))
        
        return {
            'time': t,
            'population': P_trajectory,
            'resource_deficit': D_trajectory,
            'extinction_probability': extinction_prob,
            'extinction_time': self._find_extinction_time(t, P_trajectory)
        }
    
    def _find_extinction_time(self, t: np.ndarray, P: np.ndarray, threshold: float = 10.0) -> Optional[float]:
        """Find time when population drops below threshold"""
        extinction_indices = np.where(P < threshold)[0]
        if len(extinction_indices) > 0:
            return t[extinction_indices[0]]
        return None
    
    def stability_analysis(self) -> Dict:
        """
        Analyze system stability using Laplace transform poles and zeros
        """
        # System transfer function: G(s) = K/(s + k)
        # Pole location: s = -k
        pole = -self.k
        
        # Stability metrics
        stability = {
            'pole_location': pole,
            'is_stable': pole < 0,
            'time_constant': 1/self.k,
            'settling_time_95': 3/self.k,  # 95% response time
            'settling_time_99': 5/self.k,  # 99% response time
            'bandwidth': self.k / (2*np.pi),  # -3dB bandwidth in Hz
            'steady_state_gain': 1/self.k
        }
        
        return stability
    
    def early_warning_indicators(self) -> Dict:
        """
        Calculate early warning indicators based on critical slowing down
        """
        total_H = self.cross_entropy_components['total']
        
        # Variance (should increase as system approaches critical point)
        variance = np.var(total_H)
        
        # Autocorrelation at lag-1 (should increase near critical point)
        autocorr_lag1 = np.corrcoef(total_H[:-1], total_H[1:])[0, 1]
        
        # Rate of change
        dH_dt = np.gradient(total_H, self.dt)
        rate_of_change = np.mean(np.abs(dH_dt))
        
        # Skewness (distribution asymmetry)
        from scipy.stats import skew
        skewness = skew(total_H)
        
        # Critical slowing down indicator (inverse of adaptation rate)
        slowing_indicator = 1 / self.k
        
        return {
            'variance': variance,
            'autocorrelation_lag1': autocorr_lag1,
            'rate_of_change': rate_of_change,
            'skewness': skewness,
            'critical_slowing_indicator': slowing_indicator,
            'mean_cross_entropy': np.mean(total_H),
            'max_cross_entropy': np.max(total_H)
        }
    
    def comprehensive_analysis(self, save_results: bool = True, save_dir: str = ".") -> Dict:
        """
        Run complete analysis pipeline
        """
        print("🐱 Starting Cross-Entropy Survival Theory Analysis...")
        
        # Step 1: Generate resource data
        print("📊 Generating cat resource data...")
        resources = self.generate_cat_resource_data()
        
        # Step 2: Calculate cross-entropy components
        print("🧮 Calculating cross-entropy components...")
        cross_entropy = self.calculate_cross_entropy_components()
        
        # Step 3: Fourier analysis
        print("🌊 Performing Fourier analysis...")
        fourier_results = self.fourier_analysis()
        
        # Step 4: Laplace analysis scenarios
        print("⚡ Analyzing Laplace transform scenarios...")
        laplace_scenarios = self.laplace_analysis_scenarios()
        
        # Step 5: Population dynamics
        print("📈 Simulating population dynamics...")
        pop_dynamics = {}
        for scenario_name in laplace_scenarios.keys():
            pop_dynamics[scenario_name] = self.population_dynamics_simulation(scenario_name)
        
        # Step 6: Stability analysis
        print("🔍 Performing stability analysis...")
        stability = self.stability_analysis()
        
        # Step 7: Early warning indicators
        print("⚠️ Calculating early warning indicators...")
        warnings = self.early_warning_indicators()
        
        print("✅ Analysis complete!")
        
        analysis_results = {
            'resources': resources,
            'cross_entropy': cross_entropy,
            'fourier': fourier_results,
            'laplace_scenarios': laplace_scenarios,
            'population_dynamics': pop_dynamics,
            'stability': stability,
            'early_warnings': warnings,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_parameters': {
                    'adaptation_rate': self.k,
                    'sensitivity': self.alpha,
                    'critical_threshold': self.D_critical,
                    'dt': self.dt
                }
            }
        }
        
        if save_results:
            self._save_results_to_files(analysis_results, save_dir)
        
        return analysis_results
    
    def _save_results_to_files(self, results: Dict, save_dir: str = "."):
        """
        Save analysis results to JSON and CSV files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results as JSON (excluding numpy arrays)
        json_results = self._convert_numpy_for_json(results)
        json_filename = os.path.join(save_dir, f"CEST_results_{timestamp}.json")
        with open(json_filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"📊 Results saved: {json_filename}")
        
        # Save key metrics as CSV
        csv_data = []
        for scenario, pop_data in results['population_dynamics'].items():
            csv_data.append({
                'scenario': scenario,
                'final_population': pop_data['population'][-1],
                'extinction_time': pop_data['extinction_time'],
                'min_population': np.min(pop_data['population']),
                'max_deficit': np.max(pop_data['resource_deficit'])
            })
        
        df = pd.DataFrame(csv_data)
        csv_filename = os.path.join(save_dir, f"CEST_population_summary_{timestamp}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"📈 Population summary saved: {csv_filename}")
        
        return json_filename, csv_filename
    
    def _convert_numpy_for_json(self, obj):
        """
        Convert numpy arrays and other non-serializable objects for JSON
        """
        if isinstance(obj, np.ndarray):
            # Convert numpy array to list, handling complex numbers
            return [self._convert_numpy_for_json(item) for item in obj.tolist()]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (complex, np.complex64, np.complex128)):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        elif isinstance(obj, ResourceVariable):
            return {
                'name': obj.name,
                'required_mean': float(np.mean(obj.required)),
                'available_mean': float(np.mean(obj.available)),
                'critical_threshold': obj.critical_threshold,
                'weight': obj.weight
            }
        else:
            return obj
    
    def generate_results_markdown(self, results: Dict, plot_filename: str, save_dir: str = ".") -> str:
        """
        Generate comprehensive markdown report explaining the results
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown_content = f"""# Cross-Entropy Survival Theory (CEST) Analysis Report

**Generated:** {timestamp}
**Analysis Framework:** Cross-Entropy Survival Theory with Fourier and Laplace Transform Analysis

---

## Executive Summary

This analysis applies Cross-Entropy Survival Theory to model cat population dynamics under various environmental disturbances. The theory quantifies survival probability based on resource availability mismatches using information-theoretic cross-entropy measures.

### Key Model Parameters
- **Adaptation Rate (k):** {self.k} day⁻¹ (Time constant: {1/self.k:.2f} days)
- **Sensitivity (α):** {self.alpha}
- **Critical Threshold:** {self.D_critical}
- **Time Resolution:** {self.dt} days

---

## 1. Resource Analysis

The model tracks four critical resources for cat survival:

"""
        
        # Add resource analysis
        for name, resource in results['resources'].items():
            avg_required = np.mean(resource.required)
            avg_available = np.mean(resource.available)
            deficit_ratio = (avg_required - avg_available) / avg_required * 100
            
            markdown_content += f"""### {resource.name}
- **Average Required:** {avg_required:.2f} units
- **Average Available:** {avg_available:.2f} units
- **Deficit/Surplus:** {deficit_ratio:+.1f}% {'(DEFICIT)' if deficit_ratio > 0 else '(SURPLUS)'}
- **Critical Threshold:** {resource.critical_threshold} units
- **Weight in Analysis:** {resource.weight}

"""
        
        # Add Fourier Analysis section
        markdown_content += f"""---

## 2. Fourier Analysis Results

Fourier analysis reveals the dominant periodic patterns in resource availability and cross-entropy fluctuations:

"""
        
        for name in ['protein', 'water', 'temperature', 'shelter']:
            fourier_data = results['fourier'][name]
            dominant_period = fourier_data['dominant_periods'][0]
            total_power = fourier_data['total_power']
            
            markdown_content += f"""### {name.capitalize()} Resource
- **Dominant Period:** {dominant_period:.1f} days
- **Total Spectral Power:** {total_power:.2e}
- **Pattern Interpretation:** """
            
            if 350 <= dominant_period <= 380:
                markdown_content += "Annual seasonal cycle\n"
            elif 25 <= dominant_period <= 35:
                markdown_content += "Monthly urban development cycles\n"
            elif 6 <= dominant_period <= 8:
                markdown_content += "Weekly human activity patterns\n"
            elif 0.8 <= dominant_period <= 1.2:
                markdown_content += "Daily circadian patterns\n"
            else:
                markdown_content += f"Custom cycle ({dominant_period:.1f} day period)\n"
            
            markdown_content += "\n"
        
        # Add Laplace Analysis section
        stability = results['stability']
        markdown_content += f"""---

## 3. Laplace Transform Analysis

System stability analysis using Laplace transforms:

### System Stability Metrics
- **Pole Location:** {stability['pole_location']:.3f} (Left half-plane = Stable)
- **System Status:** {'✅ STABLE' if stability['is_stable'] else '❌ UNSTABLE'}
- **Time Constant:** {stability['time_constant']:.2f} days
- **95% Settling Time:** {stability['settling_time_95']:.2f} days
- **99% Settling Time:** {stability['settling_time_99']:.2f} days

### Disturbance Scenario Analysis

"""
        
        for scenario_name, scenario_data in results['laplace_scenarios'].items():
            description = scenario_data['description']
            transfer_func = scenario_data['transfer_function']
            final_value = scenario_data['final_value']
            
            markdown_content += f"""#### {scenario_name.replace('_', ' ').title()}
- **Description:** {description}
- **Transfer Function:** {transfer_func}
- **Final State:** {final_value}
- **Interpretation:** """
            
            if scenario_name == 'step_disturbance':
                markdown_content += "Sudden food source removal leads to new equilibrium state\n"
            elif scenario_name == 'ramp_disturbance':
                markdown_content += "Gradual climate change causes unbounded deficit growth\n"
            elif scenario_name == 'impulse_disturbance':
                markdown_content += "Temporary habitat shock with eventual recovery\n"
            
            markdown_content += "\n"
        
        # Add Population Dynamics section
        markdown_content += f"""---

## 4. Population Dynamics Results

Population response to different environmental disturbances:

"""
        
        for scenario_name, pop_data in results['population_dynamics'].items():
            final_pop = pop_data['population'][-1]
            extinction_time = pop_data['extinction_time']
            min_pop = np.min(pop_data['population'])
            
            markdown_content += f"""### {scenario_name.replace('_', ' ').title()}
- **Final Population:** {final_pop:.0f} individuals
- **Minimum Population:** {min_pop:.0f} individuals
- **Extinction Time:** {'Day ' + str(int(extinction_time)) if extinction_time else 'No extinction within timeframe'}
- **Survival Outcome:** {'❌ EXTINCTION' if extinction_time else '✅ SURVIVAL'}

"""
        
        # Add Early Warning Indicators
        warnings = results['early_warnings']
        markdown_content += f"""---

## 5. Early Warning Indicators

Critical slowing down indicators for system collapse prediction:

### Statistical Indicators
- **Cross-Entropy Variance:** {warnings['variance']:.6f}
- **Autocorrelation (lag-1):** {warnings['autocorrelation_lag1']:.4f}
- **Rate of Change:** {warnings['rate_of_change']:.6f}
- **Distribution Skewness:** {warnings['skewness']:.4f}

### Critical Thresholds
- **Mean Cross-Entropy:** {warnings['mean_cross_entropy']:.4f}
- **Maximum Cross-Entropy:** {warnings['max_cross_entropy']:.4f}
- **Critical Slowing Indicator:** {warnings['critical_slowing_indicator']:.2f}

### Risk Assessment
"""
        
        if warnings['variance'] > 0.01:
            markdown_content += "⚠️ **HIGH VARIANCE** - System showing increased instability\n"
        if warnings['autocorrelation_lag1'] > 0.8:
            markdown_content += "⚠️ **HIGH AUTOCORRELATION** - System approaching critical transition\n"
        if warnings['mean_cross_entropy'] > 5.0:
            markdown_content += "🚨 **HIGH CROSS-ENTROPY** - Severe resource-requirement mismatches\n"
        
        markdown_content += f"""
---

## 6. Methodology Explanation

### Cross-Entropy Survival Theory Framework

1. **Resource Modeling:** Four critical resources (protein, water, temperature, shelter) are modeled as time series with stochastic variations

2. **Cross-Entropy Calculation:** 
   - For each resource j: H_j(t) = -p_j(t) × log(q_j(t))
   - Where p_j = required/total_required, q_j = available/total_available
   - Total cross-entropy: H(t) = Σ w_j × H_j(t)

3. **Fourier Analysis:** Identifies dominant frequencies in resource fluctuations
   - Reveals seasonal, monthly, weekly, and daily patterns
   - Power spectral density shows energy distribution across frequencies

4. **Laplace Transform Analysis:** Studies system response to disturbances
   - Step function: Sudden resource loss
   - Ramp function: Gradual environmental change  
   - Impulse function: Temporary shock events

5. **Population Dynamics:** ODE integration with cross-entropy mortality
   - dP/dt = r×P×(1-P/K) - α×P×max(0, D(t)-D_critical)
   - Links information theory to population biology

### Mathematical Foundation

The core equation linking cross-entropy to survival probability:
```
S(t) = exp(-α × max(0, H(t) - H_critical))
```

Where:
- S(t) = Survival probability at time t
- α = Species sensitivity parameter  
- H(t) = Cross-entropy measure at time t
- H_critical = Critical cross-entropy threshold

---

## 7. Visual Results

![Comprehensive Analysis Results]({os.path.basename(plot_filename)})

The comprehensive visualization above shows:
- **Top Row:** Resource availability, cross-entropy components, power spectra, total spectral power
- **Second Row:** Laplace analysis for three disturbance scenarios
- **Third Row:** Population dynamics responses to disturbances  
- **Bottom Row:** System stability analysis and early warning indicators

---

## 8. Conclusions and Implications

### Key Findings

1. **System Stability:** The cat population system is {'stable' if stability['is_stable'] else 'unstable'} with a time constant of {stability['time_constant']:.2f} days

2. **Critical Threats:** """
        
        # Determine most threatening scenario
        threat_levels = {}
        for scenario_name, pop_data in results['population_dynamics'].items():
            if pop_data['extinction_time']:
                threat_levels[scenario_name] = pop_data['extinction_time']
        
        if threat_levels:
            most_threatening = min(threat_levels, key=threat_levels.get)
            markdown_content += f"{most_threatening.replace('_', ' ').title()} poses the greatest immediate threat\n"
        else:
            markdown_content += "No immediate extinction threats detected within analysis timeframe\n"
        
        markdown_content += f"""
3. **Dominant Patterns:** Resource availability shows strong seasonal cycles ({results['fourier']['protein']['dominant_periods'][0]:.0f}-day periods)

4. **Early Warnings:** """
        
        if warnings['variance'] > 0.01 or warnings['autocorrelation_lag1'] > 0.8:
            markdown_content += "System showing early warning signs of critical transition\n"
        else:
            markdown_content += "No immediate early warning signals detected\n"
        
        markdown_content += f"""
### Conservation Recommendations

1. **Resource Management:** Focus on stabilizing {['protein', 'water', 'temperature', 'shelter'][np.argmax([results['fourier'][r]['total_power'] for r in ['protein', 'water', 'temperature', 'shelter']])]} resources (highest variability)

2. **Monitoring Strategy:** Implement {stability['settling_time_95']:.0f}-day monitoring cycles based on system response time

3. **Intervention Timing:** Act within {stability['time_constant']:.0f} days of detecting critical threshold breaches

4. **Habitat Protection:** Prioritize protection against {'gradual habitat degradation' if threat_levels and 'ramp' in most_threatening else 'sudden habitat loss'} scenarios

---

## 9. Technical Details

**Analysis Runtime:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model Implementation:** Python with SciPy, NumPy, Matplotlib
**Data Points:** {len(self.time_vector):,} time steps over {self.time_vector[-1]:.0f} days
**Computational Method:** Numerical ODE integration with FFT analysis

---

*This report was generated using Cross-Entropy Survival Theory (CEST) analysis framework. For technical details, see the accompanying code and data files.*
"""
        
        # Save markdown file
        markdown_filename = os.path.join(save_dir, f"CEST_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(markdown_filename, 'w') as f:
            f.write(markdown_content)
        
        print(f"📝 Analysis report saved: {markdown_filename}")
        return markdown_filename
    
    def create_comprehensive_plots(self, results: Dict, save_dir: str = "."):
        """
        Create comprehensive visualization of all results and save to files
        """
        fig = plt.figure(figsize=(20, 24))
        
        # Plot 1: Resource time series
        ax1 = plt.subplot(4, 3, 1)
        for name, resource in results['resources'].items():
            plt.plot(self.time_vector[:1000], resource.available[:1000], 
                    label=f'{name} available', alpha=0.7)
            plt.axhline(y=np.mean(resource.required), color='red', 
                       linestyle='--', alpha=0.5, label=f'{name} required')
        plt.title('Resource Availability vs Requirements')
        plt.xlabel('Time (days)')
        plt.ylabel('Resource Amount')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Cross-entropy components
        ax2 = plt.subplot(4, 3, 2)
        for name, H_component in results['cross_entropy'].items():
            if name != 'total':
                plt.plot(self.time_vector[:1000], H_component[:1000], 
                        label=name, alpha=0.7)
        plt.plot(self.time_vector[:1000], results['cross_entropy']['total'][:1000], 
                'k-', linewidth=2, label='Total')
        plt.title('Cross-Entropy Components Over Time')
        plt.xlabel('Time (days)')
        plt.ylabel('Cross-Entropy H(t)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Fourier power spectra
        ax3 = plt.subplot(4, 3, 3)
        for name in ['protein', 'water', 'temperature', 'shelter']:
            freqs = results['fourier'][name]['frequencies']
            psd = results['fourier'][name]['power']
            # Plot only positive frequencies
            pos_freq_mask = freqs > 0
            plt.loglog(freqs[pos_freq_mask], psd[pos_freq_mask], 
                      label=name, alpha=0.7)
        plt.title('Power Spectral Density (Fourier Analysis)')
        plt.xlabel('Frequency (1/days)')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Dominant frequencies
        ax4 = plt.subplot(4, 3, 4)
        variables = list(results['fourier'].keys())[:-1]  # Exclude 'total'
        total_powers = [results['fourier'][var]['total_power'] for var in variables]
        plt.bar(variables, total_powers, alpha=0.7)
        plt.title('Total Spectral Power by Resource Variable')
        plt.ylabel('Total Power')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 5-7: Laplace scenarios
        scenarios = ['step_disturbance', 'ramp_disturbance', 'impulse_disturbance']
        for i, scenario in enumerate(scenarios):
            ax = plt.subplot(4, 3, 5+i)
            data = results['laplace_scenarios'][scenario]
            plt.plot(data['time'], data['response'], 'b-', linewidth=2)
            plt.axhline(y=self.D_critical, color='red', linestyle='--', 
                       label=f'Critical threshold = {self.D_critical}')
            plt.title(f'Laplace Analysis: {data["description"]}')
            plt.xlabel('Time (days)')
            plt.ylabel('Resource Deficit D(t)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 8-10: Population dynamics
        for i, scenario in enumerate(scenarios):
            ax = plt.subplot(4, 3, 8+i)
            pop_data = results['population_dynamics'][scenario]
            
            # Population trajectory
            ax_pop = ax
            ax_pop.plot(pop_data['time'], pop_data['population'], 'g-', linewidth=2)
            ax_pop.set_xlabel('Time (days)')
            ax_pop.set_ylabel('Population Size', color='g')
            ax_pop.tick_params(axis='y', labelcolor='g')
            
            # Extinction probability on secondary axis
            ax_ext = ax_pop.twinx()
            ax_ext.plot(pop_data['time'], pop_data['extinction_probability'], 
                       'r--', alpha=0.7)
            ax_ext.set_ylabel('Extinction Probability', color='r')
            ax_ext.tick_params(axis='y', labelcolor='r')
            
            plt.title(f'Population Response: {scenario.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
        
        # Plot 11: Stability analysis visualization
        ax11 = plt.subplot(4, 3, 11)
        stability = results['stability']
        
        # Pole-zero plot
        plt.scatter([stability['pole_location']], [0], color='red', s=100, 
                   marker='x', label=f'Pole at s = {stability["pole_location"]:.2f}')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlim([-1, 0.5])
        plt.ylim([-0.5, 0.5])
        plt.title('System Pole Location (Stability Analysis)')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add stability text
        stability_text = (f"Time Constant: {stability['time_constant']:.2f} days\n"
                         f"Settling Time (95%): {stability['settling_time_95']:.2f} days\n"
                         f"System {'STABLE' if stability['is_stable'] else 'UNSTABLE'}")
        plt.text(0.02, 0.98, stability_text, transform=ax11.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Plot 12: Early warning indicators
        ax12 = plt.subplot(4, 3, 12)
        warnings = results['early_warnings']
        
        indicators = ['variance', 'autocorrelation_lag1', 'rate_of_change', 'skewness']
        values = [warnings[ind] for ind in indicators]
        colors = ['blue', 'green', 'orange', 'purple']
        
        bars = plt.bar(indicators, values, color=colors, alpha=0.7)
        plt.title('Early Warning Indicators')
        plt.ylabel('Indicator Value')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add warning text
        warning_text = (f"Mean Cross-Entropy: {warnings['mean_cross_entropy']:.3f}\n"
                       f"Max Cross-Entropy: {warnings['max_cross_entropy']:.3f}\n"
                       f"Critical Slowing: {warnings['critical_slowing_indicator']:.2f}")
        plt.text(0.02, 0.98, warning_text, transform=ax12.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral'))
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_filename = os.path.join(save_dir, "CEST_comprehensive_analysis.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📈 Comprehensive plot saved: {plot_filename}")
        
        plt.show()
        
        return fig, plot_filename

def run_complete_analysis():
    """
    Run the complete Cross-Entropy Survival Theory analysis
    """
    # Initialize model with realistic parameters for cats
    model = CrossEntropySurvivalModel(
        adaptation_rate=0.3,      # Cats adapt relatively quickly (3.3 day time constant)
        sensitivity=0.5,          # Moderate sensitivity to resource mismatches
        critical_threshold=8.0,   # Critical cross-entropy threshold
        dt=0.1                    # High temporal resolution (2.4 hours)
    )
    
    # Run comprehensive analysis
    results = model.comprehensive_analysis()
    
    # Print key findings
    print("\n" + "="*60)
    print("🎯 KEY FINDINGS - CROSS-ENTROPY SURVIVAL THEORY")
    print("="*60)
    
    print(f"\n📊 FOURIER ANALYSIS RESULTS:")
    print("-" * 30)
    for name in ['protein', 'water', 'temperature', 'shelter']:
        fourier_data = results['fourier'][name]
        dominant_period = fourier_data['dominant_periods'][0]
        print(f"{name.capitalize():12}: Dominant period = {dominant_period:.1f} days")
    
    print(f"\n⚡ LAPLACE ANALYSIS RESULTS:")
    print("-" * 30)
    stability = results['stability']
    print(f"System Stability: {'STABLE' if stability['is_stable'] else 'UNSTABLE'}")
    print(f"Time Constant:    {stability['time_constant']:.2f} days")
    print(f"Settling Time:    {stability['settling_time_95']:.2f} days (95%)")
    
    print(f"\n🚨 POPULATION DYNAMICS:")
    print("-" * 30)
    for scenario_name, pop_data in results['population_dynamics'].items():
        extinction_time = pop_data['extinction_time']
        final_population = pop_data['population'][-1]
        print(f"{scenario_name.replace('_', ' ').title():20}: "
              f"Final pop = {final_population:.0f}, "
              f"Extinction = {'Yes' if extinction_time else 'No'}")
    
    print(f"\n⚠️ EARLY WARNING INDICATORS:")
    print("-" * 30)
    warnings = results['early_warnings']
    print(f"Cross-Entropy Variance:     {warnings['variance']:.4f}")
    print(f"Autocorrelation (lag-1):    {warnings['autocorrelation_lag1']:.4f}")
    print(f"Mean Cross-Entropy:         {warnings['mean_cross_entropy']:.4f}")
    print(f"Critical Slowing Indicator: {warnings['critical_slowing_indicator']:.2f}")
    
    # Create comprehensive plots and generate report
    print(f"\n📈 Generating comprehensive visualization...")
    fig, plot_filename = model.create_comprehensive_plots(results)
    
    # Generate markdown report
    print(f"\n📝 Generating analysis report...")
    markdown_filename = model.generate_results_markdown(results, plot_filename)
    
    return model, results, plot_filename, markdown_filename

if __name__ == "__main__":
    # Run the complete analysis
    model, results, plot_filename, markdown_filename = run_complete_analysis()
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📊 Results saved in current directory:")
    print(f"   • Plot: {os.path.basename(plot_filename)}")
    print(f"   • Report: {os.path.basename(markdown_filename)}")
    print(f"   • Data files: CEST_results_*.json and CEST_population_summary_*.csv")
    print(f"💡 This implementation demonstrates the full Cross-Entropy Survival Theory")
    print(f"   with both Fourier and Laplace transform analysis techniques.")
