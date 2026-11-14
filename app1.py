from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import customtkinter as ctk
import pandas as pd
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import joblib
except ImportError:
    joblib = None


# ----------------------------- THEME ---------------------------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

C = {
    "bg": "#F8F1F7",
    "surface": "#FFFFFF",
    "surface_alt": "#FFF9FD",
    "sidebar": "#32143D",
    "sidebar_hover": "#4B2057",
    "sidebar_active": "#6B3475",
    "primary": "#9B4F91",
    "primary_hover": "#7F3D78",
    "orchid": "#D98ACB",
    "orchid_light": "#F0D7EC",
    "text": "#321B35",
    "muted": "#806D82",
    "border": "#E8D8E8",
    "success": "#477B68",
    "success_bg": "#E8F3EE",
    "danger": "#A94768",
    "danger_bg": "#F8E8EF",
    "warning": "#A87535",
    "warning_bg": "#FBF1DF",
}

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "processed" / "supply_chain_processed.csv"
METRICS_PATH = ROOT / "models" / "model_comparison.csv"
MODEL_PATH = ROOT / "models" / "random_forest_model.pkl"
HIGH_RISK_PATH = ROOT / "reports" / "high_risk_orders.csv"

TARGET = "Late_delivery_risk"
NUMERIC_FIELDS = [
    ("Days for shipment (scheduled)", "4"),
    ("Benefit per order", "50"),
    ("Sales per customer", "150"),
    ("Order Item Discount", "15"),
    ("Order Item Discount Rate", "0.05"),
    ("Order Item Quantity", "1"),
    ("Sales", "165"),
    ("Order Item Total", "150"),
    ("Order Profit Per Order", "25"),
    ("Product Price", "165"),
]
CATEGORICAL_OPTIONS = {
    "Type": ["CASH", "DEBIT", "PAYMENT", "TRANSFER"],
    "Customer Segment": ["Consumer", "Corporate", "Home Office"],
    "Market": ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"],
    "Shipping Mode": ["First Class", "Same Day", "Second Class", "Standard Class"],
}
CAT_COLUMNS = list(CATEGORICAL_OPTIONS)


def load_csv(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_float(value, field_name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Enter a valid number for {field_name}.")
    if pd.isna(number):
        raise ValueError(f"{field_name} cannot be empty.")
    return number


class SupplyChainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Supply Chain Control Tower")
        self.geometry("1380x880")
        self.minsize(1120, 720)
        self.configure(fg_color=C["bg"])

        self.processed = load_csv(DATA_PATH)
        self.metrics = load_csv(METRICS_PATH)
        self.high_risk = load_csv(HIGH_RISK_PATH)
        self.model = None
        self.feature_columns = []
        self.nav_buttons = {}
        self.current_page = "`Dashboard`"

        self._configure_grid()
        self._build_sidebar()
        self._build_workspace()
        self.show_page("Dashboard")

    def _configure_grid(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self, width=248, corner_radius=0, fg_color=C["sidebar"]
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(8, weight=1)

        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=22, pady=(30, 28))

        mark = ctk.CTkLabel(
            brand, text="✦", width=42, height=42, corner_radius=13,
            fg_color=C["orchid"], text_color=C["sidebar"],
            font=ctk.CTkFont(size=25, weight="bold")
        )
        mark.pack(anchor="w", pady=(0, 14))

        ctk.CTkLabel(
            brand, text="SUPPLY CHAIN", text_color="#FFFFFF",
            font=ctk.CTkFont(size=17, weight="bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand, text="CONTROL TOWER", text_color=C["orchid"],
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(1, 0))
        ctk.CTkLabel(
            brand, text="RISK INTELLIGENCE PLATFORM",
            text_color="#C9B3CC", font=ctk.CTkFont(size=9, weight="bold")
        ).pack(anchor="w", pady=(9, 0))

        ctk.CTkLabel(
            self.sidebar, text="WORKSPACE", text_color="#B99ABE",
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(anchor="w", padx=24, pady=(5, 10))

        pages = [
            ("Dashboard", "⌂"),
            ("Risk Prediction", "◎"),
            ("What-If Analysis", "⇄"),
            ("High-Risk Monitoring", "▤"),
        ]
        for page, symbol in pages:
            button = ctk.CTkButton(
                self.sidebar, text=f"   {symbol}    {page}",
                anchor="w", height=46, corner_radius=11,
                fg_color="transparent", hover_color=C["sidebar_hover"],
                text_color="#F8EDF8",
                font=ctk.CTkFont(size=13, weight="bold"),
                command=lambda p=page: self.show_page(p)
            )
            button.pack(fill="x", padx=14, pady=4)
            self.nav_buttons[page] = button

        ctk.CTkLabel(
            self.sidebar, text="Build ",
            text_color="#A98AAD", font=ctk.CTkFont(size=9)
        ).pack(side="bottom", anchor="w", padx=22, pady=20)

    def _build_workspace(self):
        self.workspace = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        self.workspace.grid(row=0, column=1, sticky="nsew", padx=30, pady=25)
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(2, weight=1)

        top = ctk.CTkFrame(self.workspace, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        self.page_title = ctk.CTkLabel(
            top, text="Dashboard", text_color=C["text"],
            font=ctk.CTkFont(size=29, weight="bold")
        )
        self.page_title.grid(row=0, column=0, sticky="w")

        self.date_label = ctk.CTkLabel(
            top, text="SUPPLY CHAIN • RISK MONITORING",
            text_color=C["muted"], font=ctk.CTkFont(size=10, weight="bold")
        )
        self.date_label.grid(row=1, column=0, sticky="w", pady=(5, 0))

        self.content = ctk.CTkScrollableFrame(
            self.workspace, fg_color="transparent", corner_radius=0
        )
        self.content.grid(row=2, column=0, sticky="nsew", pady=(24, 0))
        self.content.grid_columnconfigure(0, weight=1)

    def _clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def _section_title(self, parent, title, subtitle=None):
        ctk.CTkLabel(
            parent, text=title, text_color=C["text"],
            font=ctk.CTkFont(size=17, weight="bold")
        ).pack(anchor="w")
        if subtitle:
            ctk.CTkLabel(
                parent, text=subtitle, text_color=C["muted"],
                font=ctk.CTkFont(size=11)
            ).pack(anchor="w", pady=(4, 0))

    def _card(self, parent, title, value, note="", accent=None):
        frame = ctk.CTkFrame(
            parent, fg_color=C["surface"], corner_radius=16,
            border_width=1, border_color=C["border"]
        )
        ctk.CTkFrame(
            frame, height=4, corner_radius=3,
            fg_color=accent or C["orchid"]
        ).pack(fill="x", padx=16, pady=(14, 12))
        ctk.CTkLabel(
            frame, text=title.upper(), text_color=C["muted"],
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(anchor="w", padx=18)
        ctk.CTkLabel(
            frame, text=value, text_color=C["text"],
            font=ctk.CTkFont(size=25, weight="bold")
        ).pack(anchor="w", padx=18, pady=(5, 2))
        if note:
            ctk.CTkLabel(
                frame, text=note, text_color=C["muted"],
                font=ctk.CTkFont(size=10)
            ).pack(anchor="w", padx=18, pady=(0, 15))
        else:
            ctk.CTkFrame(frame, height=15, fg_color="transparent").pack()
        return frame

    def _add_chart(self, parent, title, draw_function, height=3.0):
        wrapper = ctk.CTkFrame(
            parent, fg_color=C["surface"], corner_radius=16,
            border_width=1, border_color=C["border"]
        )
        ctk.CTkLabel(
            wrapper, text=title, text_color=C["text"],
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=17, pady=(15, 0))

        if not MATPLOTLIB_AVAILABLE:
            ctk.CTkLabel(
                wrapper, text="Install matplotlib to display charts.",
                text_color=C["muted"]
            ).pack(padx=15, pady=25)
            return wrapper

        fig = Figure(figsize=(5.2, height), dpi=95, facecolor=C["surface"])
        ax = fig.add_subplot(111)
        draw_function(ax)
        ax.set_facecolor(C["surface"])
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=wrapper)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        return wrapper

    def _dataset_stats(self):
        if self.processed is None or TARGET not in self.processed.columns:
            return None
        counts = self.processed[TARGET].value_counts().to_dict()
        total = len(self.processed)
        high = int(counts.get(1, 0))
        low = int(counts.get(0, 0))
        return total, high, low

    def show_page(self, page):
        self.current_page = page
        self.page_title.configure(text=page)
        self._clear_content()
        for name, button in self.nav_buttons.items():
            button.configure(
                fg_color=C["sidebar_active"] if name == page else "transparent"
            )

        pages = {
            "Dashboard": self._dashboard,
            "Risk Prediction": self._prediction_page,
            "What-If Analysis": self._what_if_page,
            "High-Risk Monitoring": self._high_risk_page,
        }
        pages[page]()

    # ----------------------------- DASHBOARD -------------------------------
      
    # ----------------------------- DASHBOARD -------------------------------
    def _dashboard(self):
        """Display actual dataset statistics and model performance."""

        intro = ctk.CTkFrame(self.content, fg_color="transparent")
        intro.pack(fill="x", pady=(0, 17))

        ctk.CTkLabel(
            intro,
            text="Monitor delivery risks and supply chain performance",
            text_color=C["muted"],
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w")

        ctk.CTkLabel(
            intro,
            text="PROJECT OVERVIEW  /  MODEL PERFORMANCE",
            text_color=C["primary"],
            font=ctk.CTkFont(size=9, weight="bold")
        ).pack(anchor="w", pady=(7, 0))

        # Get actual dataset statistics
        stats = self._dataset_stats()

        total = high = low = 0
        high_pct = low_pct = 0

        if stats is not None:
            total, high, low = stats

            if total > 0:
                high_pct = (high / total) * 100
                low_pct = (low / total) * 100

        # Get model comparison information
        metrics_available = (
            self.metrics is not None
            and all(
                column in self.metrics.columns
                for column in [
                    "Model", "Accuracy", "Precision", "Recall", "F1-Score"
                ]
            )
        )

        model_count = len(self.metrics) if metrics_available else 0
        best_accuracy_note = "Model comparison data unavailable"

        if metrics_available and model_count > 0:
            accuracy_values = pd.to_numeric(
                self.metrics["Accuracy"], errors="coerce"
            )

            if accuracy_values.notna().any():
                best_index = accuracy_values.idxmax()
                best_model = self.metrics.loc[best_index, "Model"]
                best_accuracy = accuracy_values.loc[best_index]

                if best_accuracy <= 1:
                    best_accuracy *= 100

                best_accuracy_note = (
                    f"Best: {best_model} ({best_accuracy:.2f}%)"
                )

        # KPI cards
        kpi_area = ctk.CTkFrame(
            self.content, fg_color="transparent"
        )
        kpi_area.pack(fill="x", pady=(0, 18))

        for col in range(4):
            kpi_area.grid_columnconfigure(
                col, weight=1, uniform="kpis"
            )

        kpis = [
            (
                "TOTAL ORDERS",
                f"{total:,}" if stats is not None else "N/A",
                "Orders in processed dataset",
                C["primary"],
                "▦"
            ),
            (
                "HIGH DELIVERY RISK",
                f"{high:,}" if stats is not None else "N/A",
                f"{high_pct:.2f}% of total orders",
                C["danger"],
                "↑"
            ),
            (
                "LOW DELIVERY RISK",
                f"{low:,}" if stats is not None else "N/A",
                f"{low_pct:.2f}% of total orders",
                C["success"],
                "✓"
            ),
            (
                "MODEL COUNT",
                str(model_count) if metrics_available else "N/A",
                best_accuracy_note,
                C["orchid"],
                "✦"
            ),
        ]

        for index, (label, value, note, accent, symbol) in enumerate(kpis):

            card = ctk.CTkFrame(
                kpi_area,
                fg_color=C["surface"],
                corner_radius=15,
                border_width=1,
                border_color=C["border"]
            )

            card.grid(
                row=0,
                column=index,
                sticky="nsew",
                padx=(0 if index == 0 else 7, 0)
            )

            ctk.CTkFrame(
                card,
                height=4,
                fg_color=accent,
                corner_radius=3
            ).pack(fill="x", padx=14, pady=(13, 11))

            top_line = ctk.CTkFrame(
                card, fg_color="transparent"
            )
            top_line.pack(fill="x", padx=15)

            ctk.CTkLabel(
                top_line,
                text=label,
                text_color=C["muted"],
                font=ctk.CTkFont(size=9, weight="bold")
            ).pack(side="left", anchor="w")

            ctk.CTkLabel(
                top_line,
                text=symbol,
                text_color=accent,
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(side="right")

            ctk.CTkLabel(
                card,
                text=value,
                text_color=C["text"],
                font=ctk.CTkFont(size=25, weight="bold")
            ).pack(anchor="w", padx=15, pady=(7, 1))

            ctk.CTkLabel(
                card,
                text=note,
                text_color=C["muted"],
                font=ctk.CTkFont(size=9),
                wraplength=210
            ).pack(anchor="w", padx=15, pady=(0, 13))

        # Chart layout
        chart_row = ctk.CTkFrame(
            self.content, fg_color="transparent"
        )
        chart_row.pack(fill="x")

        chart_row.grid_columnconfigure(
            (0, 1), weight=1, uniform="dashboard_charts"
        )

        # Model performance comparison chart
        def model_metrics_chart(ax):
            ax.clear()

            if not metrics_available or self.metrics.empty:
                ax.text(
                    0.5, 0.5,
                    "Model comparison data unavailable",
                    ha="center", va="center",
                    color=C["muted"]
                )
                ax.set_axis_off()
                return

            metric_columns = [
                "Accuracy", "Precision", "Recall", "F1-Score"
            ]

            chart_data = self.metrics.copy()

            for column in metric_columns:
                chart_data[column] = pd.to_numeric(
                    chart_data[column], errors="coerce"
                )

            chart_data = chart_data.dropna(
                subset=metric_columns
            )

            if chart_data.empty:
                ax.text(
                    0.5, 0.5,
                    "No valid model metrics found",
                    ha="center", va="center",
                    color=C["muted"]
                )
                ax.set_axis_off()
                return

            # Convert decimal scores into percentages if necessary
            if chart_data[metric_columns].max().max() <= 1:
                chart_data[metric_columns] *= 100

            models = chart_data["Model"].astype(str).tolist()

            x = list(range(len(models)))
            bar_width = 0.18

            colors = [
                C["primary"],
                C["orchid"],
                C["success"],
                C["warning"]
            ]

            for index, metric in enumerate(metric_columns):
                values = chart_data[metric].tolist()

                positions = [
                    position + (index - 1.5) * bar_width
                    for position in x
                ]

                ax.bar(
                    positions,
                    values,
                    bar_width,
                    label=metric,
                    color=colors[index]
                )

            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=8)
            ax.set_ylabel("Score (%)", fontsize=8)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.tick_params(axis="y", labelsize=8)

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.25
            )

            ax.set_axisbelow(True)

            ax.legend(
                fontsize=7,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=2,
                frameon=False
            )

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        # High-risk and low-risk distribution chart
        def risk_distribution_chart(ax):
            ax.clear()

            if stats is None or total == 0:
                ax.text(
                    0.5, 0.5,
                    "Dataset statistics unavailable",
                    ha="center", va="center",
                    color=C["muted"]
                )
                ax.set_axis_off()
                return

            if high + low == 0:
                ax.text(
                    0.5, 0.5,
                    "No risk records available",
                    ha="center", va="center",
                    color=C["muted"]
                )
                ax.set_axis_off()
                return

            ax.pie(
                [high, low],
                labels=["High Risk", "Low Risk"],
                colors=[C["danger"], C["success"]],
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.78,
                wedgeprops={
                    "width": 0.42,
                    "edgecolor": C["surface"]
                },
                textprops={"fontsize": 9}
            )

            ax.text(
                0, 0,
                f"{total:,}\nOrders",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=C["text"]
            )

            ax.axis("equal")

        # Display both charts
        model_card = self._add_chart(
            chart_row,
            "Model performance comparison",
            model_metrics_chart,
            height=3.0
        )
        model_card.grid(
            row=0, column=0, sticky="nsew", padx=(0, 8)
        )

        risk_card = self._add_chart(
            chart_row,
            "Late-delivery risk distribution",
            risk_distribution_chart,
            height=3.0
        )
        risk_card.grid(
            row=0, column=1, sticky="nsew", padx=(8, 0)
        )
        
        # Global Model Feature Importance
        importance_card = ctk.CTkFrame(
            self.content,
            fg_color=C["surface"],
            corner_radius=15,
            border_width=1,
            border_color=C["border"]
        )
        importance_card.pack(fill="x", pady=(16, 12))

        importance_header = ctk.CTkFrame(
            importance_card,
            fg_color="transparent"
        )
        importance_header.pack(
            fill="x",
            padx=16,
            pady=(14, 5)
        )

        ctk.CTkLabel(
            importance_header,
            text="GLOBAL FEATURE IMPORTANCE",
            text_color=C["primary"],
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left")

        model_selector = ctk.CTkComboBox(
            importance_header,
            values=[
                "Random Forest",
                "Logistic Regression",
                "Decision Tree"
            ],
            width=190,
            height=34,
            fg_color=C["surface"],
            border_color=C["border"],
            button_color=C["primary"],
            button_hover_color=C["primary_hover"],
            text_color=C["text"]
        )
        model_selector.set("Random Forest")
        model_selector.pack(side="right")

        importance_figure = Figure(
            figsize=(10, 4.2),
            dpi=100
        )

        importance_ax = importance_figure.add_subplot(111)

        importance_canvas = FigureCanvasTkAgg(
            importance_figure,
            master=importance_card
        )

        importance_canvas.get_tk_widget().pack(
            fill="x",
            padx=15,
            pady=(5, 15)
        )

        def update_importance_chart(choice=None):
            importance_ax.clear()

            if not self.feature_columns:
                self._load_model()

            importance_data = self._get_global_feature_importance(
                model_selector.get()
            )

            if importance_data.empty:
                importance_ax.text(
                    0.5,
                    0.5,
                    "Feature importance data unavailable",
                    ha="center",
                    va="center",
                    transform=importance_ax.transAxes
                )
                importance_ax.set_axis_off()

            else:
                chart_data = importance_data.sort_values(
                    "Importance",
                    ascending=True
                )

                importance_ax.barh(
                    chart_data["Feature"],
                    chart_data["Importance"],
                    color=C["primary"]
                )

                importance_ax.set_title(
                    f"Top 10 Features - {model_selector.get()}",
                    fontsize=11,
                    fontweight="bold"
                )

                importance_ax.set_xlabel(
                    "Importance Score",
                    fontsize=9
                )

                importance_ax.tick_params(
                    axis="both",
                    labelsize=8
                )

                importance_ax.grid(
                    axis="x",
                    linestyle="--",
                    alpha=0.25
                )

                importance_ax.set_axisbelow(True)

                for spine in ["top", "right"]:
                    importance_ax.spines[spine].set_visible(False)

            importance_figure.tight_layout()
            importance_canvas.draw()

        model_selector.configure(
            command=update_importance_chart
        )

        update_importance_chart()



    # ----------------------------- MODEL HELPERS ---------------------------
    def _load_model(self):
        if self.model is not None:
            return self.model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "The saved model 'models/random_forest_model.pkl' is not available."
            )
        if joblib is None:
            raise ImportError("joblib is required to load the saved model.")
        self.model = joblib.load(MODEL_PATH)
        if self.processed is None:
            raise FileNotFoundError("processed/supply_chain_processed.csv is missing.")
        self.feature_columns = [c for c in self.processed.columns if c != TARGET]
        return self.model
    
    def _get_global_feature_importance(self, model_name):
        """Extract global feature importance from a trained model."""

        model_paths = {
            "Random Forest": ROOT / "models" / "random_forest_model.pkl",
            "Logistic Regression": ROOT / "models" / "logistic_regression_model.pkl",
            "Decision Tree": ROOT / "models" / "decision_tree_model.pkl"
        }

        try:
            if joblib is None:
                raise ImportError("Joblib is unavailable.")

            model = joblib.load(model_paths[model_name])

            if model_name == "Logistic Regression":
                classifier = model.named_steps["logisticregression"]
                importance = np.abs(classifier.coef_[0])

            else:
                importance = model.feature_importances_

            if len(importance) != len(self.feature_columns):
                raise ValueError(
                    "Feature importance count does not match model features."
                )

            importance_df = pd.DataFrame({
                "Feature": self.feature_columns,
                "Importance": importance
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            return importance_df.head(10)

        except Exception as exc:
            print(f"Feature importance error ({model_name}): {exc}")
            return pd.DataFrame(columns=["Feature", "Importance"])

    def _prepare_model_input(self, order):
        """    """
        self._load_model()
        # Start with numeric inputs, then set the matching saved one-hot columns.
        # This avoids creating different dummy columns for a single-row prediction.
        encoded = pd.DataFrame([{key: value for key, value in order.items()
                                 if key not in CAT_COLUMNS}])
        for column in self.feature_columns:
            if column not in encoded.columns:
                encoded[column] = 0
        for category in CAT_COLUMNS:
            selected_value = str(order.get(category, ""))
            dummy_column = f"{category}_{selected_value}"
            if dummy_column in self.feature_columns:
                encoded.loc[0, dummy_column] = 1
        return encoded.reindex(columns=self.feature_columns, fill_value=0)

    
    def _predict(self, order):
        model_input = self._prepare_model_input(order)
        prediction = int(self.model.predict(model_input)[0])
        probabilities = self.model.predict_proba(model_input)[0]
        classes = list(self.model.classes_)
        high_index = classes.index(1) if 1 in classes else len(classes) - 1
        return prediction, float(probabilities[high_index])

    def _explain_prediction(self, model_input, prediction):
        if not SHAP_AVAILABLE:
            return "SHAP is not installed. Explanation is unavailable."

        try:
            model = self._load_model()
            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(
                model_input,
                check_additivity=False
            )

            if hasattr(shap_values, "values"):
                shap_values = shap_values.values

            classes = list(model.classes_)
            high_index = classes.index(1)

            if isinstance(shap_values, list):
                contributions = np.asarray(
                    shap_values[high_index]
                )[0]

            else:
                shap_values = np.asarray(shap_values)

                if shap_values.ndim == 3:
                    if shap_values.shape[-1] == len(classes):
                        contributions = shap_values[0, :, high_index]
                    elif shap_values.shape[0] == len(classes):
                        contributions = shap_values[high_index, 0, :]
                    else:
                        raise ValueError(
                            "Unexpected SHAP output dimensions."
                        )

                elif shap_values.ndim == 2:
                    contributions = shap_values[0]

                else:
                    raise ValueError(
                        "Unsupported SHAP output format."
                    )

            explanation = pd.DataFrame({
                "Feature": model_input.columns,
                "Feature Value": model_input.iloc[0].values,
                "Contribution": contributions
            })

            increasing = explanation[
                explanation["Contribution"] > 0
            ].sort_values("Contribution", ascending=False)

            decreasing = explanation[
                explanation["Contribution"] < 0
            ].sort_values("Contribution", ascending=True)

            def format_feature(row):
                feature = str(row["Feature"])
                value = row["Feature Value"]

                for category in CAT_COLUMNS:
                    prefix = category + "_"

                    if feature.startswith(prefix):
                        selected = (
                            "selected" if float(value) == 1
                            else "not selected"
                        )

                        return (
                            f"{category} = "
                            f"{feature[len(prefix):]} ({selected})"
                        )

                return f"{feature} (value: {value})"

            lines = []

            if prediction == 1:
                lines.append("WHY IS THE RISK HIGH?")
                lines.append(
                    "Factors increasing the predicted risk:"
                )

                main_factors = increasing
                opposite_factors = decreasing
                main_label = "Increases risk"
                opposite_label = "Reduces risk"

            else:
                lines.append("WHY IS THE RISK LOW?")
                lines.append(
                    "Factors reducing the predicted risk:"
                )

                main_factors = decreasing
                opposite_factors = increasing
                main_label = "Reduces risk"
                opposite_label = "Increases risk"

            if main_factors.empty:
                lines.append(
                    "No clear supporting factors were identified."
                )
            else:
                for _, row in main_factors.head(3).iterrows():
                    lines.append(
                        f"• {format_feature(row)} — {main_label}"
                    )

            if not opposite_factors.empty:
                lines.append("")
                lines.append(
                    "Factors working in the opposite direction:"
                )

                for _, row in opposite_factors.head(2).iterrows():
                    lines.append(
                        f"• {format_feature(row)} — {opposite_label}"
                    )

            lines.append("")
            lines.append(
                "Note: These are model contributions, "
                "not causal explanations."
            )

            return "\n".join(lines)

        except Exception as exc:
            return f"Explanation unavailable: {exc}"

    def _collect_order(self, numeric_entries, categorical_boxes):
        order = {}
        for field, entry in numeric_entries.items():
            order[field] = safe_float(entry.get(), field)
        for field, combo in categorical_boxes.items():
            order[field] = combo.get()
        return order

    def _make_form_field(self, parent, label, default, row, col, width=220):
        ctk.CTkLabel(
            parent, text=label, text_color=C["muted"],
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=row*2, column=col, sticky="w", padx=12, pady=(10, 3))
        entry = ctk.CTkEntry(
            parent, width=width, height=38, corner_radius=9,
            fg_color=C["surface_alt"], border_color=C["border"],
            text_color=C["text"], placeholder_text=label
        )
        entry.insert(0, default)
        entry.grid(row=row*2+1, column=col, sticky="ew", padx=12, pady=(0, 6))
        return entry

    def _make_combo_field(self, parent, label, values, default, row, col):
        ctk.CTkLabel(
            parent, text=label, text_color=C["muted"],
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=row*2, column=col, sticky="w", padx=12, pady=(10, 3))
        combo = ctk.CTkComboBox(
            parent, values=values, height=38, corner_radius=9,
            fg_color=C["surface_alt"], border_color=C["border"],
            button_color=C["primary"], button_hover_color=C["primary_hover"],
            text_color=C["text"], dropdown_fg_color=C["surface"],
            dropdown_text_color=C["text"], dropdown_hover_color=C["orchid_light"]
        )
        combo.set(default)
        combo.grid(row=row*2+1, column=col, sticky="ew", padx=12, pady=(0, 6))
        return combo


    
    # ----------------------------- RISK PREDICTION -------------------------
    def _prediction_page(self):
        self._section_title(
            self.content,
            "Predict delivery risk",
            "Enter order details to estimate the probability of late delivery."
        )

        if not MODEL_PATH.exists():
            notice = ctk.CTkFrame(
                self.content,
                fg_color=C["warning_bg"],
                corner_radius=12
            )
            notice.pack(fill="x", pady=(14, 10))

            ctk.CTkLabel(
                notice,
                text=(
                    "Trained model not found. The form is available, "
                    "but prediction requires models/random_forest_model.pkl."
                ),
                text_color=C["warning"],
                wraplength=900,
                justify="left",
                font=ctk.CTkFont(size=11)
            ).pack(anchor="w", padx=15, pady=12)

        form = ctk.CTkFrame(
            self.content,
            fg_color=C["surface"],
            corner_radius=16,
            border_width=1,
            border_color=C["border"]
        )
        form.pack(fill="x", pady=(14, 12))

        for col in range(2):
            form.grid_columnconfigure(col, weight=1)

        numeric_entries = {}

        for index, (field, default) in enumerate(NUMERIC_FIELDS):
            numeric_entries[field] = self._make_form_field(
                form,
                field,
                default,
                index // 2,
                index % 2
            )

        categorical_boxes = {}

        defaults = {
            "Type": "DEBIT",
            "Customer Segment": "Consumer",
            "Market": "Africa",
            "Shipping Mode": "Standard Class"
        }

        offset = len(NUMERIC_FIELDS) // 2

        for index, (field, options) in enumerate(CATEGORICAL_OPTIONS.items()):
            categorical_boxes[field] = self._make_combo_field(
                form,
                field,
                options,
                defaults[field],
                offset + index // 2,
                index % 2
            )

        # Prediction result card
        result = ctk.CTkFrame(
            self.content,
            fg_color=C["surface"],
            corner_radius=16,
            border_width=1,
            border_color=C["border"]
        )
        result.pack(fill="x", pady=(4, 12))

        result_label = ctk.CTkLabel(
            result,
            text="Your prediction will appear here.",
            text_color=C["muted"],
            font=ctk.CTkFont(size=13),
            justify="left",
            wraplength=850
        )
        result_label.pack(anchor="w", padx=18, pady=18)

        # Run prediction and generate SHAP explanation
        def run_prediction():
            try:
                order = self._collect_order(
                    numeric_entries,
                    categorical_boxes
                )

                if order["Days for shipment (scheduled)"] <= 0:
                    raise ValueError(
                        "Scheduled shipping days must be greater than zero."
                    )

                if not 0 <= order["Order Item Discount Rate"] <= 0.25:
                    raise ValueError(
                        "Discount rate must be between 0 and 0.25."
                    )

                if not 1 <= order["Order Item Quantity"] <= 5:
                    raise ValueError(
                        "Order quantity must be between 1 and 5."
                    )

                for key in [
                    "Order Item Discount",
                    "Sales",
                    "Order Item Total",
                    "Product Price"
                ]:
                    if order[key] < 0:
                        raise ValueError(
                            f"{key} cannot be negative."
                        )

                prediction, high_probability = self._predict(order)

                risk = "HIGH" if prediction == 1 else "LOW"

                color = (
                    C["danger"] if prediction == 1
                    else C["success"]
                )

                model_input = self._prepare_model_input(order)

                explanation = self._explain_prediction(
                    model_input,
                    prediction
                )

                result_label.configure(
                    text=(
                        f"PREDICTED DELIVERY RISK: {risk}\n"
                        f"High-risk probability: "
                        f"{high_probability * 100:.2f}%\n"
                        f"Low-risk probability: "
                        f"{(1 - high_probability) * 100:.2f}%\n\n"
                        f"{explanation}"
                    ),
                    text_color=color
                )

            except Exception as exc:
                result_label.configure(
                    text=f"Prediction unavailable: {exc}",
                    text_color=C["warning"]
                )

        # Prediction button
        ctk.CTkButton(
            self.content,
            text="Predict Delivery Risk",
            height=42,
            fg_color=C["primary"],
            hover_color=C["primary_hover"],
            font=ctk.CTkFont(size=13, weight="bold"),
            command=run_prediction
        ).pack(anchor="w", pady=(0, 16))

    # ----------------------------- WHAT-IF ANALYSIS ------------------------

    # ----------------------------- WHAT-IF ANALYSIS ------------------------
    def _what_if_page(self):
        """Compare original and modified delivery-risk predictions."""
        self._section_title(
            self.content,
            "What-If Scenario Analysis",
            "Compare delivery risk by changing shipping mode or scheduled shipment days."
        )

        # ---------------- ORIGINAL ORDER ----------------
        original_card = ctk.CTkFrame(
            self.content, fg_color=C["surface"], corner_radius=15,
            border_width=1, border_color=C["border"]
        )
        original_card.pack(fill="x", pady=(14, 12))
        original_card.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            original_card, text="1. ORIGINAL ORDER DETAILS",
            text_color=C["primary"],
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w",
               padx=14, pady=(14, 4))

        ctk.CTkLabel(
            original_card,
            text="Enter the starting order details for comparison.",
            text_color=C["muted"], font=ctk.CTkFont(size=10)
        ).grid(row=1, column=0, columnspan=2, sticky="w",
               padx=14, pady=(0, 5))

        original_defaults = {
            "Days for shipment (scheduled)": "4",
            "Benefit per order": "25",
            "Sales per customer": "150",
            "Order Item Discount": "10",
            "Order Item Discount Rate": "0.10",
            "Order Item Quantity": "2",
            "Sales": "150",
            "Order Item Total": "140",
            "Order Profit Per Order": "25",
            "Product Price": "75"
        }

        numeric_entries = {}

        for index, (field, _default) in enumerate(NUMERIC_FIELDS):
            numeric_entries[field] = self._make_form_field(
                original_card,
                field,
                original_defaults[field],
                index // 2 + 1,
                index % 2
            )

        category_defaults = {
            "Type": "DEBIT",
            "Customer Segment": "Consumer",
            "Market": "Africa",
            "Shipping Mode": "Standard Class"
        }

        category_boxes = {}
        category_row = 6

        for index, field in enumerate(CAT_COLUMNS):
            category_boxes[field] = self._make_combo_field(
                original_card,
                field,
                CATEGORICAL_OPTIONS[field],
                category_defaults[field],
                category_row + index // 2,
                index % 2
            )

        # ---------------- MODIFIED SCENARIO ----------------
        modified_card = ctk.CTkFrame(
            self.content, fg_color=C["surface"], corner_radius=15,
            border_width=1, border_color=C["border"]
        )
        modified_card.pack(fill="x", pady=(0, 12))
        modified_card.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            modified_card, text="2. MODIFIED SCENARIO",
            text_color=C["primary"],
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w",
               padx=14, pady=(14, 4))

        ctk.CTkLabel(
            modified_card,
            text="Change shipping mode and/or scheduled shipment days. "
                 "All other order details remain unchanged.",
            text_color=C["muted"], font=ctk.CTkFont(size=10),
            wraplength=850, justify="left"
        ).grid(row=1, column=0, columnspan=2, sticky="w",
               padx=14, pady=(0, 8))

        ctk.CTkLabel(
            modified_card,
            text="Modified shipping mode",
            text_color=C["muted"],
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=2, column=0, sticky="w", padx=14, pady=(5, 3))

        modified_mode = ctk.CTkComboBox(
            modified_card,
            values=CATEGORICAL_OPTIONS["Shipping Mode"],
            height=38,
            fg_color=C["surface_alt"],
            border_color=C["border"],
            button_color=C["primary"],
            button_hover_color=C["primary_hover"],
            text_color=C["text"]
        )
        modified_mode.set("Same Day")
        modified_mode.grid(row=3, column=0, sticky="ew",
                           padx=14, pady=(0, 13))

        ctk.CTkLabel(
            modified_card,
            text="Modified scheduled shipment days",
            text_color=C["muted"],
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=2, column=1, sticky="w", padx=14, pady=(5, 3))

        modified_days = ctk.CTkEntry(
            modified_card,
            height=38,
            fg_color=C["surface_alt"],
            border_color=C["border"],
            text_color=C["text"]
        )
        modified_days.insert(0, "2")
        modified_days.grid(row=3, column=1, sticky="ew",
                           padx=14, pady=(0, 13))

        # ---------------- PREDICTION RESULTS ----------------
        ctk.CTkLabel(
            self.content,
            text="3. ORIGINAL VS MODIFIED PREDICTION",
            text_color=C["text"],
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", pady=(5, 8))

        result_row = ctk.CTkFrame(
            self.content, fg_color="transparent"
        )
        result_row.pack(fill="x", pady=(0, 10))
        result_row.grid_columnconfigure((0, 1), weight=1)

        def result_card(parent, title):
            card = ctk.CTkFrame(
                parent,
                fg_color=C["surface"],
                corner_radius=13,
                border_width=1,
                border_color=C["border"]
            )

            ctk.CTkLabel(
                card,
                text=title,
                text_color=C["primary"],
                font=ctk.CTkFont(size=11, weight="bold")
            ).pack(anchor="w", padx=15, pady=(13, 5))

            value = ctk.CTkLabel(
                card,
                text="Prediction: —\nHigh-risk probability: —",
                text_color=C["muted"],
                justify="left",
                font=ctk.CTkFont(size=12)
            )
            value.pack(anchor="w", padx=15, pady=(3, 15))

            return card, value

        original_result_card, original_result = result_card(
            result_row, "ORIGINAL ORDER"
        )
        modified_result_card, modified_result = result_card(
            result_row, "MODIFIED ORDER"
        )

        original_result_card.grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        modified_result_card.grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )

        difference_card = ctk.CTkFrame(
            self.content,
            fg_color=C["surface_alt"],
            corner_radius=12,
            border_width=1,
            border_color=C["border"]
        )
        difference_card.pack(fill="x", pady=(0, 10))

        difference_label = ctk.CTkLabel(
            difference_card,
            text="Risk difference: —",
            text_color=C["muted"],
            font=ctk.CTkFont(size=12, weight="bold")
        )
        difference_label.pack(anchor="w", padx=15, pady=13)

        # ---------------- COMPARISON TABLE ----------------
        table_card = ctk.CTkFrame(
            self.content,
            fg_color=C["surface"],
            corner_radius=14,
            border_width=1,
            border_color=C["border"]
        )
        table_card.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            table_card,
            text="SCENARIO COMPARISON TABLE",
            text_color=C["primary"],
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=14, pady=(12, 7))

        columns = (
            "Scenario",
            "Shipping Mode",
            "Scheduled Days",
            "Predicted Risk",
            "High-Risk Score",
            "Risk Difference"
        )

        table = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings",
            height=3
        )

        for column in columns:
            table.heading(column, text=column)
            table.column(
                column, width=145, minwidth=110, anchor="center"
            )

        table.pack(fill="x", padx=12, pady=(0, 12))

        status = ctk.CTkLabel(
            self.content,
            text="Enter the order details and click Compare Scenarios.",
            text_color=C["muted"],
            wraplength=950,
            justify="left",
            font=ctk.CTkFont(size=11)
        )
        status.pack(anchor="w", pady=(0, 16))

        # ---------------- VALIDATION ----------------
        def validate_order(order):
            if order["Days for shipment (scheduled)"] <= 0:
                raise ValueError(
                    "Scheduled shipment days must be greater than zero."
                )

            if not 0 <= order["Order Item Discount Rate"] <= 0.25:
                raise ValueError(
                    "Discount rate must be between 0 and 0.25."
                )

            if not 1 <= order["Order Item Quantity"] <= 5:
                raise ValueError(
                    "Order quantity must be between 1 and 5."
                )

            for key in [
                "Benefit per order",
                "Sales per customer",
                "Order Item Discount",
                "Sales",
                "Order Item Total",
                "Order Profit Per Order",
                "Product Price"
            ]:
                if order[key] < 0:
                    raise ValueError(f"{key} cannot be negative.")

            for field in CAT_COLUMNS:
                if order[field] not in CATEGORICAL_OPTIONS[field]:
                    raise ValueError(f"Select a valid value for {field}.")

        # ---------------- RUN WHAT-IF PREDICTIONS ----------------
        def prepare_comparison():
            try:
                # Collect original order details
                original = self._collect_order(
                    numeric_entries, category_boxes
                )

                validate_order(original)

                # Create a copy so original details are not modified
                modified = original.copy()

                new_days = safe_float(
                    modified_days.get(),
                    "Modified scheduled shipment days"
                )

                if new_days <= 0:
                    raise ValueError(
                        "Modified scheduled shipment days must be greater than zero."
                    )

                new_shipping = modified_mode.get()

                if new_shipping not in CATEGORICAL_OPTIONS["Shipping Mode"]:
                    raise ValueError("Select a valid modified shipping mode.")

                # Change only the two scenario fields
                modified["Days for shipment (scheduled)"] = new_days
                modified["Shipping Mode"] = new_shipping

                # Get predictions from the trained Random Forest model
                original_prediction, original_probability = self._predict(
                    original
                )

                modified_prediction, modified_probability = self._predict(
                    modified
                )

                original_risk = (
                    "HIGH" if original_prediction == 1 else "LOW"
                )
                modified_risk = (
                    "HIGH" if modified_prediction == 1 else "LOW"
                )

                # Calculate difference in percentage points
                difference = (
                    modified_probability - original_probability
                ) * 100

                # Update original result card
                original_result.configure(
                    text=(
                        f"Shipping mode: {original['Shipping Mode']}\n"
                        f"Scheduled days: "
                        f"{original['Days for shipment (scheduled)']:g}\n"
                        f"Prediction: {original_risk}\n"
                        f"High-risk probability: "
                        f"{original_probability * 100:.2f}%"
                    ),
                    text_color=(
                        C["danger"] if original_prediction == 1
                        else C["success"]
                    )
                )

                # Update modified result card
                modified_result.configure(
                    text=(
                        f"Shipping mode: {new_shipping}\n"
                        f"Scheduled days: {new_days:g}\n"
                        f"Prediction: {modified_risk}\n"
                        f"High-risk probability: "
                        f"{modified_probability * 100:.2f}%"
                    ),
                    text_color=(
                        C["danger"] if modified_prediction == 1
                        else C["success"]
                    )
                )

                # Update risk difference
                if difference > 0.005:
                    difference_text = (
                        f"Risk increased by {difference:.2f} percentage points."
                    )
                    difference_color = C["danger"]

                elif difference < -0.005:
                    difference_text = (
                        f"Risk decreased by {abs(difference):.2f} percentage points."
                    )
                    difference_color = C["success"]

                else:
                    difference_text = (
                        "No significant change in high-risk probability."
                    )
                    difference_color = C["muted"]

                difference_label.configure(
                    text=(
                        f"Risk difference: {difference:+.2f} percentage points\n"
                        f"{difference_text}"
                    ),
                    text_color=difference_color
                )

                # Refresh comparison table
                for item in table.get_children():
                    table.delete(item)

                table.insert(
                    "",
                    "end",
                    values=(
                        "Original",
                        original["Shipping Mode"],
                        f"{original['Days for shipment (scheduled)']:g}",
                        original_risk,
                        f"{original_probability * 100:.2f}%",
                        "Baseline"
                    )
                )

                table.insert(
                    "",
                    "end",
                    values=(
                        "Modified",
                        new_shipping,
                        f"{new_days:g}",
                        modified_risk,
                        f"{modified_probability * 100:.2f}%",
                        f"{difference:+.2f} pp"
                    )
                )

                # Final scenario message
                if difference > 0.005:
                    status_message = (
                        "The modified scenario increases the predicted "
                        "high-risk probability."
                    )
                elif difference < -0.005:
                    status_message = (
                        "The modified scenario reduces the predicted "
                        "high-risk probability."
                    )
                else:
                    status_message = (
                        "The modified scenario has almost the same "
                        "predicted high-risk probability."
                    )

                status.configure(
                    text=status_message,
                    text_color=difference_color
                )

            except Exception as exc:
                messagebox.showerror(
                    "What-If Prediction Error",
                    str(exc)
                )

        ctk.CTkButton(
            self.content,
            text="Compare Scenarios",
            height=42,
            fg_color=C["primary"],
            hover_color=C["primary_hover"],
            command=prepare_comparison
        ).pack(anchor="w", pady=(0, 16))

    # ----------------------------- HIGH-RISK MONITORING --------------------
    
    
    def _high_risk_page(self):
        """Show the High-Risk Monitoring layout."""

        self._section_title(
            self.content,
            "High-Risk Order Monitoring",
            "Review high-risk orders, available order details, and model risk scores."
        )

        # Search and filter controls
        toolbar = ctk.CTkFrame(self.content, fg_color="transparent")
        toolbar.pack(fill="x", pady=(0, 10))

        search_entry = ctk.CTkEntry(
            toolbar,
            width=300,
            height=38,
            placeholder_text="Search order details...",
            fg_color=C["surface"],
            border_color=C["border"],
            text_color=C["text"]
        )
        search_entry.pack(side="left", padx=(0, 8))

        risk_filter = ctk.CTkComboBox(
            toolbar,
            values=["All risks", "HIGH", "LOW"],
            width=145,
            height=38,
            fg_color=C["surface"],
            border_color=C["border"],
            button_color=C["primary"],
            button_hover_color=C["primary_hover"],
            text_color=C["text"]
        )
        risk_filter.set("All risks")
        risk_filter.pack(side="left")

        refresh_button = ctk.CTkButton(
            toolbar,
            text="Refresh report",
            height=38,
            width=130,
            fg_color=C["primary"],
            hover_color=C["primary_hover"]
        )
        refresh_button.pack(side="left", padx=8)

        # Table card
        table_card = ctk.CTkFrame(
            self.content,
            fg_color=C["surface"],
            corner_radius=15,
            border_width=1,
            border_color=C["border"]
        )
        table_card.pack(fill="both", expand=True, pady=(0, 12))
        table_card.grid_columnconfigure(0, weight=1)
        table_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            table_card,
            text="HIGH-RISK ORDERS TABLE",
            text_color=C["primary"],
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(13, 8))

        columns = (
            "Actual Risk",
            "Predicted Risk",
            "High-Risk Probability (%)",
            "Shipping Mode",
            "Scheduled Shipment Days",
            "Sales",
            "Order Profit Per Order"
        )

        tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings",
            height=10
        )

        widths = {
            "Actual Risk": 110,
            "Predicted Risk": 120,
            "High-Risk Probability (%)": 190,
            "Shipping Mode": 150,
            "Scheduled Shipment Days": 190,
            "Sales": 120,
            "Order Profit Per Order": 190
        }

        for column in columns:
            tree.heading(column, text=column)
            tree.column(
                column,
                width=widths[column],
                minwidth=110,
                anchor="center"
            )

        yscroll = ttk.Scrollbar(
            table_card,
            orient="vertical",
            command=tree.yview
        )

        xscroll = ttk.Scrollbar(
            table_card,
            orient="horizontal",
            command=tree.xview
        )

        tree.configure(
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set
        )

        tree.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(0, 4))
        yscroll.grid(row=1, column=1, sticky="ns", pady=(0, 4), padx=(0, 12))
        xscroll.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))

        # Table styling
        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Treeview",
            background="#FFF9FD",
            foreground=C["text"],
            fieldbackground="#FFF9FD",
            rowheight=32,
            bordercolor=C["border"]
        )

        style.configure(
            "Treeview.Heading",
            background=C["orchid_light"],
            foreground=C["text"],
            font=("Segoe UI", 9, "bold")
        )

        style.map(
            "Treeview",
            background=[("selected", C["orchid_light"])]
        )

        tree.tag_configure("HIGH", foreground="#D9485F")
        tree.tag_configure("LOW", foreground="#16865B")

        # Empty state
        empty_state = ctk.CTkLabel(
            table_card,
            text="Loading high-risk report...",
            text_color=C["muted"],
            justify="center",
            font=ctk.CTkFont(size=14)
        )
        empty_state.place(relx=0.5, rely=0.57, anchor="center")

        # Selected order details
        details_card = ctk.CTkFrame(
            self.content,
            fg_color=C["surface"],
            corner_radius=14,
            border_width=1,
            border_color=C["border"]
        )
        details_card.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            details_card,
            text="SELECTED ORDER DETAILS",
            text_color=C["primary"],
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(12, 4))

        details_label = ctk.CTkLabel(
            details_card,
            text="Select an order from the table to view its details.",
            text_color=C["muted"],
            justify="left",
            wraplength=1000
        )
        details_label.pack(anchor="w", padx=15, pady=(0, 13))
        explanation_label = ctk.CTkLabel(
            details_card,
            text="Select an order to view its risk explanation.",
            text_color=C["muted"],
            justify="left",
            wraplength=1000,
            font=ctk.CTkFont(size=12)
        )
        explanation_label.pack(
            anchor="w",
            padx=15,
            pady=(0, 15)
        )

        # Find shipping mode
        def get_shipping_mode(row):
            if row.get("Shipping Mode_Same Day", 0) == 1:
                return "Same Day"

            if row.get("Shipping Mode_Second Class", 0) == 1:
                return "Second Class"

            if row.get("Shipping Mode_Standard Class", 0) == 1:
                return "Standard Class"

            return "First Class"

        # Display report records
        def refresh_table(*args):
            tree.delete(*tree.get_children())

            if self.high_risk is None or self.high_risk.empty:
                empty_state.configure(
                    text="No high-risk report data available."
                )
                return

            filtered_data = self.high_risk.copy()

            # Risk filter
            selected_risk = risk_filter.get()

            if selected_risk in ["HIGH", "LOW"]:
                filtered_data = filtered_data[
                    filtered_data["Predicted_Risk"] == selected_risk
                ]

            # Search
            search_text = search_entry.get().strip().lower()

            if search_text:
                mask = filtered_data.astype(str).apply(
                    lambda column: column.str.lower().str.contains(
                        search_text,
                        regex=False,
                        na=False
                    )
                ).any(axis=1)

                filtered_data = filtered_data[mask]

            if filtered_data.empty:
                empty_state.configure(text="No matching orders found.")
                details_label.configure(text="No order selected.")
                explanation_label.configure(
                    text="No order available for explanation.",
                    text_color=C["muted"]
                )
                return

            empty_state.configure(text="")

            for index, row in filtered_data.iterrows():

                actual_risk = (
                    "HIGH" if int(row["Actual_Risk"]) == 1 else "LOW"
                )

                predicted_risk = row["Predicted_Risk"]
                probability = float(row["High_Risk_Probability"])

                shipping_mode = get_shipping_mode(row)
                scheduled_days = row["Days for shipment (scheduled)"]
                sales = float(row["Sales"])
                profit = float(row["Order Profit Per Order"])

                values = (
                    actual_risk,
                    predicted_risk,
                    f"{probability:.2f}%",
                    shipping_mode,
                    scheduled_days,
                    f"{sales:.2f}",
                    f"{profit:.2f}"
                )

                tree.insert(
                    "",
                    "end",
                    iid=str(index),
                    values=values,
                    tags=(predicted_risk,)
                )

        # Selected order details
        def show_selected_order(event=None):
            selected = tree.selection()

            if not selected:
                return

            try:
                index = int(selected[0])
                row = self.high_risk.loc[index]

                actual_risk = (
                    "HIGH" if int(row["Actual_Risk"]) == 1 else "LOW"
                )

                predicted_risk = str(row["Predicted_Risk"]).upper()
                probability = float(row["High_Risk_Probability"])

                shipping_mode = get_shipping_mode(row)

                details_text = (
                    f"Actual Risk: {actual_risk}     "
                    f"Predicted Risk: {predicted_risk}     "
                    f"High-Risk Probability: {probability:.2f}%\n"
                    f"Shipping Mode: {shipping_mode}     "
                    f"Scheduled Days: {row['Days for shipment (scheduled)']}     "
                    f"Sales: {float(row['Sales']):.2f}     "
                    f"Order Profit: {float(row['Order Profit Per Order']):.2f}"
                )

                details_label.configure(
                    text=details_text,
                    text_color=C["text"]
                )

                # Generate SHAP explanation for selected order
                self._load_model()

                missing_columns = [
                    column for column in self.feature_columns
                    if column not in row.index
                ]

                if missing_columns:
                    raise ValueError(
                        f"Missing model features: {missing_columns}"
                    )

                model_input = row[
                    self.feature_columns
                ].to_frame().T

                prediction = (
                    1 if predicted_risk == "HIGH" else 0
                )

                explanation = self._explain_prediction(
                    model_input,
                    prediction
                )

                explanation_label.configure(
                    text=explanation,
                    text_color=C["text"]
                )

            except Exception as exc:
                explanation_label.configure(
                    text=f"Explanation unavailable: {exc}",
                    text_color=C["warning"]
                )

        # Reload report
        def refresh_report():
            try:
                self.high_risk = pd.read_csv(HIGH_RISK_PATH)
                refresh_table()

                details_label.configure(
                    text="Report refreshed. Select an order to view details."
                )

            except Exception as e:
                empty_state.configure(
                    text=f"Unable to load report:\n{e}"
                )

        # Connect controls
        search_entry.bind("<KeyRelease>", refresh_table)

        risk_filter.configure(
            command=lambda choice: refresh_table()
        )

        refresh_button.configure(command=refresh_report)

        tree.bind("<<TreeviewSelect>>", show_selected_order)

        # Load report
        refresh_table()



if __name__ == "__main__":
    app = SupplyChainApp()
    app.mainloop()