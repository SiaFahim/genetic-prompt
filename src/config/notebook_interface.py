"""
Jupyter notebook interface for hyperparameter configuration.

This module provides an intuitive interface for modifying hyperparameters
in Jupyter notebooks with real-time validation and visual feedback.
"""

import json
from typing import Dict, Any, Optional, List, Tuple
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import pandas as pd

from .hyperparameters import HyperparameterConfig, get_hyperparameter_config, set_hyperparameter_config
from .config_manager import ConfigurationManager, get_config_manager


class NotebookHyperparameterInterface:
    """Interactive interface for hyperparameter configuration in Jupyter notebooks."""
    
    def __init__(self):
        """Initialize the notebook interface."""
        self.config_manager = get_config_manager()
        self.current_config = get_hyperparameter_config()
        self.widgets = {}
        self.output_widget = widgets.Output()
        
    def create_parameter_widgets(self, category: Optional[str] = None) -> widgets.Widget:
        """
        Create interactive widgets for hyperparameter configuration.
        
        Args:
            category: Specific category to show (None for all categories)
            
        Returns:
            Widget container with parameter controls
        """
        specs = HyperparameterConfig.get_parameter_specs()
        
        if category:
            specs = {k: v for k, v in specs.items() if v.category == category}
        
        # Group by category
        categories = {}
        for param_name, spec in specs.items():
            if spec.category not in categories:
                categories[spec.category] = {}
            categories[spec.category][param_name] = spec
        
        # Create widgets for each category
        category_widgets = []
        
        for cat_name, cat_specs in categories.items():
            cat_widgets = []
            
            # Category header
            cat_header = widgets.HTML(f"<h3>📊 {cat_name.title()} Parameters</h3>")
            cat_widgets.append(cat_header)
            
            # Parameter widgets
            for param_name, spec in cat_specs.items():
                current_value = getattr(self.current_config, param_name)
                
                # Create appropriate widget based on parameter type
                if spec.parameter_type == bool:
                    widget = widgets.Checkbox(
                        value=current_value,
                        description=spec.name,
                        disabled=False,
                        style={'description_width': 'initial'}
                    )
                elif spec.parameter_type == int:
                    widget = widgets.IntSlider(
                        value=current_value,
                        min=spec.min_value or 0,
                        max=spec.max_value or 1000,
                        step=1,
                        description=spec.name,
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(width='500px')
                    )
                elif spec.parameter_type == float:
                    widget = widgets.FloatSlider(
                        value=current_value,
                        min=spec.min_value or 0.0,
                        max=spec.max_value or 1.0,
                        step=0.01,
                        description=spec.name,
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(width='500px')
                    )
                else:
                    widget = widgets.Text(
                        value=str(current_value) if current_value is not None else '',
                        description=spec.name,
                        style={'description_width': 'initial'}
                    )
                
                # Add description tooltip
                widget.tooltip = spec.description
                
                # Store widget reference
                self.widgets[param_name] = widget
                
                # Create info box
                info_html = f"""
                <div style="margin-left: 20px; font-size: 12px; color: #666;">
                    <strong>Description:</strong> {spec.description}<br>
                    <strong>Type:</strong> {spec.parameter_type.__name__}
                """
                
                if spec.min_value is not None:
                    info_html += f"<br><strong>Min:</strong> {spec.min_value}"
                if spec.max_value is not None:
                    info_html += f"<br><strong>Max:</strong> {spec.max_value}"
                if spec.valid_values is not None:
                    info_html += f"<br><strong>Valid values:</strong> {spec.valid_values}"
                
                info_html += f"<br><strong>Default:</strong> {spec.default_value}</div>"
                
                info_widget = widgets.HTML(info_html)
                
                cat_widgets.extend([widget, info_widget])
            
            # Create category container
            category_box = widgets.VBox(cat_widgets)
            category_widgets.append(category_box)
        
        # Create main container
        main_container = widgets.VBox(category_widgets)
        
        return main_container
    
    def create_control_panel(self) -> widgets.Widget:
        """Create control panel with apply, reset, and save buttons."""
        
        # Apply button
        apply_button = widgets.Button(
            description='Apply Changes',
            button_style='success',
            tooltip='Apply current parameter values',
            icon='check'
        )
        
        # Reset button
        reset_button = widgets.Button(
            description='Reset to Defaults',
            button_style='warning',
            tooltip='Reset all parameters to default values',
            icon='refresh'
        )
        
        # Save preset button
        save_button = widgets.Button(
            description='Save as Preset',
            button_style='info',
            tooltip='Save current configuration as a preset',
            icon='save'
        )
        
        # Load preset dropdown
        presets = self.config_manager.get_available_presets()
        preset_dropdown = widgets.Dropdown(
            options=list(presets.keys()),
            description='Load Preset:',
            style={'description_width': 'initial'}
        )
        
        load_button = widgets.Button(
            description='Load',
            button_style='primary',
            tooltip='Load selected preset',
            icon='upload'
        )
        
        # Status display
        status_html = widgets.HTML("<div style='color: green;'>✅ Ready</div>")
        
        # Event handlers
        def on_apply_clicked(b):
            try:
                self.apply_changes()
                status_html.value = "<div style='color: green;'>✅ Changes applied successfully</div>"
            except Exception as e:
                status_html.value = f"<div style='color: red;'>❌ Error: {str(e)}</div>"
        
        def on_reset_clicked(b):
            try:
                self.reset_to_defaults()
                status_html.value = "<div style='color: green;'>✅ Reset to defaults</div>"
            except Exception as e:
                status_html.value = f"<div style='color: red;'>❌ Error: {str(e)}</div>"
        
        def on_save_clicked(b):
            try:
                preset_name = f"custom_{len(presets)}"
                self.save_as_preset(preset_name)
                status_html.value = f"<div style='color: green;'>✅ Saved as preset: {preset_name}</div>"
            except Exception as e:
                status_html.value = f"<div style='color: red;'>❌ Error: {str(e)}</div>"
        
        def on_load_clicked(b):
            try:
                self.load_preset(preset_dropdown.value)
                status_html.value = f"<div style='color: green;'>✅ Loaded preset: {preset_dropdown.value}</div>"
            except Exception as e:
                status_html.value = f"<div style='color: red;'>❌ Error: {str(e)}</div>"
        
        apply_button.on_click(on_apply_clicked)
        reset_button.on_click(on_reset_clicked)
        save_button.on_click(on_save_clicked)
        load_button.on_click(on_load_clicked)
        
        # Layout
        button_row1 = widgets.HBox([apply_button, reset_button, save_button])
        button_row2 = widgets.HBox([preset_dropdown, load_button])
        
        control_panel = widgets.VBox([
            widgets.HTML("<h3>🎛️ Control Panel</h3>"),
            button_row1,
            button_row2,
            status_html
        ])
        
        return control_panel
    
    def apply_changes(self):
        """Apply current widget values to the configuration."""
        updates = {}
        
        for param_name, widget in self.widgets.items():
            if hasattr(widget, 'value'):
                updates[param_name] = widget.value
        
        # Update configuration
        self.current_config.update_parameters(updates)
        set_hyperparameter_config(self.current_config)
    
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        self.current_config = HyperparameterConfig()
        set_hyperparameter_config(self.current_config)
        
        # Update widget values
        for param_name, widget in self.widgets.items():
            if hasattr(widget, 'value'):
                default_value = getattr(self.current_config, param_name)
                widget.value = default_value
    
    def load_preset(self, preset_name: str):
        """Load a preset configuration."""
        self.current_config = self.config_manager.load_preset(preset_name)
        set_hyperparameter_config(self.current_config)
        
        # Update widget values
        for param_name, widget in self.widgets.items():
            if hasattr(widget, 'value'):
                preset_value = getattr(self.current_config, param_name)
                widget.value = preset_value
    
    def save_as_preset(self, preset_name: str):
        """Save current configuration as a preset."""
        self.apply_changes()  # Ensure current values are applied
        self.config_manager.save_config(self.current_config, preset_name)
    
    def display_current_config(self) -> widgets.Widget:
        """Display current configuration as a formatted table."""
        config_dict = self.current_config.to_dict()
        
        # Group by category
        specs = HyperparameterConfig.get_parameter_specs()
        categories = {}
        
        for param_name, value in config_dict.items():
            if param_name in specs:
                spec = specs[param_name]
                if spec.category not in categories:
                    categories[spec.category] = []
                categories[spec.category].append({
                    'Parameter': param_name,
                    'Value': value,
                    'Description': spec.description
                })
        
        # Create HTML table
        html_content = "<h3>📋 Current Configuration</h3>"
        
        for category, params in categories.items():
            html_content += f"<h4>{category.title()}</h4>"
            html_content += "<table style='border-collapse: collapse; width: 100%;'>"
            html_content += "<tr style='background-color: #f2f2f2;'>"
            html_content += "<th style='border: 1px solid #ddd; padding: 8px;'>Parameter</th>"
            html_content += "<th style='border: 1px solid #ddd; padding: 8px;'>Value</th>"
            html_content += "<th style='border: 1px solid #ddd; padding: 8px;'>Description</th>"
            html_content += "</tr>"
            
            for param in params:
                html_content += "<tr>"
                html_content += f"<td style='border: 1px solid #ddd; padding: 8px;'>{param['Parameter']}</td>"
                html_content += f"<td style='border: 1px solid #ddd; padding: 8px;'><strong>{param['Value']}</strong></td>"
                html_content += f"<td style='border: 1px solid #ddd; padding: 8px;'>{param['Description']}</td>"
                html_content += "</tr>"
            
            html_content += "</table><br>"
        
        return widgets.HTML(html_content)
    
    def create_full_interface(self) -> widgets.Widget:
        """Create the complete hyperparameter interface."""
        
        # Title
        title = widgets.HTML("<h1>🧬 Genetic Algorithm Hyperparameter Configuration</h1>")
        
        # Create tabs for different categories
        categories = self.current_config.get_all_categories()
        tab_children = []
        tab_titles = []
        
        for category in categories:
            category_widget = self.create_parameter_widgets(category)
            tab_children.append(category_widget)
            tab_titles.append(category.title())
        
        # Add "All Parameters" tab
        all_params_widget = self.create_parameter_widgets()
        tab_children.append(all_params_widget)
        tab_titles.append("All Parameters")
        
        # Add "Current Config" tab
        config_display = self.display_current_config()
        tab_children.append(config_display)
        tab_titles.append("Current Config")
        
        # Create tabs
        tabs = widgets.Tab(children=tab_children)
        for i, title in enumerate(tab_titles):
            tabs.set_title(i, title)
        
        # Control panel
        control_panel = self.create_control_panel()
        
        # Main interface
        main_interface = widgets.VBox([
            title,
            control_panel,
            tabs,
            self.output_widget
        ])
        
        return main_interface


# Global interface instance
_global_interface: Optional[NotebookHyperparameterInterface] = None


def get_notebook_interface() -> NotebookHyperparameterInterface:
    """Get the global notebook interface instance."""
    global _global_interface
    if _global_interface is None:
        _global_interface = NotebookHyperparameterInterface()
    return _global_interface


def display_hyperparameter_interface():
    """Display the hyperparameter configuration interface in a Jupyter notebook."""
    interface = get_notebook_interface()
    return interface.create_full_interface()


def quick_config_panel():
    """Display a quick configuration panel with most common parameters."""
    interface = get_notebook_interface()
    
    # Create widgets for most common parameters
    common_params = [
        'population_size', 'max_generations', 'crossover_rate', 'mutation_rate',
        'elite_size', 'target_fitness', 'max_problems'
    ]
    
    widgets_list = []
    config = get_hyperparameter_config()
    specs = HyperparameterConfig.get_parameter_specs()
    
    for param_name in common_params:
        if param_name in specs:
            spec = specs[param_name]
            current_value = getattr(config, param_name)
            
            if spec.parameter_type == int:
                widget = widgets.IntSlider(
                    value=current_value,
                    min=spec.min_value or 0,
                    max=spec.max_value or 1000,
                    description=spec.name,
                    style={'description_width': 'initial'}
                )
            elif spec.parameter_type == float:
                widget = widgets.FloatSlider(
                    value=current_value,
                    min=spec.min_value or 0.0,
                    max=spec.max_value or 1.0,
                    step=0.01,
                    description=spec.name,
                    style={'description_width': 'initial'}
                )
            
            widget.tooltip = spec.description
            widgets_list.append(widget)
            interface.widgets[param_name] = widget
    
    # Apply button
    apply_button = widgets.Button(
        description='Apply Quick Config',
        button_style='success',
        icon='check'
    )
    
    def on_apply_clicked(b):
        interface.apply_changes()
        print("✅ Quick configuration applied!")
    
    apply_button.on_click(on_apply_clicked)
    
    # Layout
    quick_panel = widgets.VBox([
        widgets.HTML("<h3>⚡ Quick Configuration Panel</h3>"),
        *widgets_list,
        apply_button
    ])
    
    return quick_panel
