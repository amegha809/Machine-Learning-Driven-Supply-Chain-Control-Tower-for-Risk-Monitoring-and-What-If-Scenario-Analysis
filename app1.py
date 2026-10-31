from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import customtkinter as ctk
import pandas as pd

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
        self.current_page = "Dashboard"

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

        ctk.CTkFrame(self.sidebar, height=1, fg_color="#593461").pack(
            fill="x", padx=20, pady=(22, 14)
        )
        ctk.CTkLabel(
            self.sidebar, text="PROJECT STATUS",
            text_color="#B99ABE", font=ctk.CTkFont(size=10, weight="bold")
        ).pack(anchor="w", padx=24, pady=(0, 8))

        status = "Dataset connected" if self.processed is not None else "Dataset not found"
        status_color = C["orchid"] if self.processed is not None else "#E9A7BC"
        ctk.CTkLabel(
            self.sidebar, text=f"●  {status}", text_color=status_color,
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=24)

        ctk.CTkLabel(
            self.sidebar, text="GUI BUILD  •  ORCHID EDITION",
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

        self.header_status = ctk.CTkLabel(
            top, text="●  LIVE DATA VIEW", text_color=C["success"],
            fg_color=C["success_bg"], corner_radius=18,
            font=ctk.CTkFont(size=10, weight="bold"), padx=13, pady=8
        )
        self.header_status.grid(row=0, column=1, rowspan=2, sticky="e")

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
    def _dashboard(self):
        """Dashboard only: KPIs and charts use the project's saved CSV files."""
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

        stats = self._dataset_stats()
        if stats:
            total, high, low = stats
            high_pct = (high / total * 100) if total else 0
            low_pct = (low / total * 100) if total else 0
        else:
            total = high = low = 0
            high_pct = low_pct = 0

        # Use saved comparison data; do not assume or invent model scores.
        metrics_available = (
            self.metrics is not None
            and {"Model", "Accuracy", "Precision", "Recall", "F1-Score"}.issubset(self.metrics.columns)
        )
        best_accuracy_text = "—"
        best_accuracy_note = "Saved model metrics unavailable"
        if metrics_available and not self.metrics.empty:
            best_row = self.metrics.loc[self.metrics["Accuracy"].astype(float).idxmax()]
            best_accuracy_text = f"{float(best_row['Accuracy']) * 100:.2f}%"
            best_accuracy_note = f"Highest accuracy • {best_row['Model']}"

        kpi_area = ctk.CTkFrame(self.content, fg_color="transparent")
        kpi_area.pack(fill="x", pady=(0, 18))
        for col in range(4):
            kpi_area.grid_columnconfigure(col, weight=1, uniform="kpis")

        kpis = [
            ("TOTAL ORDERS", f"{total:,}" if stats else "—",
             "Records in processed dataset" if stats else "Processed dataset not found",
             C["primary"], "▦"),
            ("HIGH DELIVERY RISK", f"{high:,}" if stats else "—",
             f"{high_pct:.1f}% of all orders" if stats else "Risk labels unavailable",
             C["danger"], "↑"),
            ("LOW DELIVERY RISK", f"{low:,}" if stats else "—",
             f"{low_pct:.1f}% of all orders" if stats else "Risk labels unavailable",
             C["success"], "✓"),
            ("MODEL COUNT", str(len(self.metrics)) if metrics_available else "—",
             best_accuracy_note if metrics_available else "Comparison file not found",
             C["orchid"], "✦"),
        ]

        for index, (label, value, note, accent, symbol) in enumerate(kpis):
            card = ctk.CTkFrame(
                kpi_area, fg_color=C["surface"], corner_radius=15,
                border_width=1, border_color=C["border"]
            )
            card.grid(row=0, column=index, sticky="nsew",
                      padx=(0 if index == 0 else 7, 0))
            ctk.CTkFrame(
                card, height=4, fg_color=accent, corner_radius=3
            ).pack(fill="x", padx=14, pady=(13, 11))
            top_line = ctk.CTkFrame(card, fg_color="transparent")
            top_line.pack(fill="x", padx=15)
            ctk.CTkLabel(
                top_line, text=label, text_color=C["muted"],
                font=ctk.CTkFont(size=9, weight="bold")
            ).pack(side="left", anchor="w")
            ctk.CTkLabel(
                top_line, text=symbol, text_color=accent,
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(side="right")
            ctk.CTkLabel(
                card, text=value, text_color=C["text"],
                font=ctk.CTkFont(size=25, weight="bold")
            ).pack(anchor="w", padx=15, pady=(7, 1))
            ctk.CTkLabel(
                card, text=note, text_color=C["muted"],
                font=ctk.CTkFont(size=9), wraplength=210
            ).pack(anchor="w", padx=15, pady=(0, 13))

        # Main chart row: model metric comparison and risk distribution.
        chart_row = ctk.CTkFrame(self.content, fg_color="transparent")
        chart_row.pack(fill="x")
        chart_row.grid_columnconfigure((0, 1), weight=1, uniform="dashboard_charts")

        def model_metrics_chart(ax):
            ax.clear()
            if not metrics_available or self.metrics.empty:
                ax.text(
                    0.5, 0.5, "Model comparison data not available",
                    ha="center", va="center", color=C["muted"], fontsize=9
                )
                ax.axis("off")
                return

            metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
            model_names = self.metrics["Model"].astype(str).tolist()
            x_positions = list(range(len(metric_names)))
            bar_width = 0.32
            model_colors = [C["primary"], C["orchid"]]

            for model_index, (_, row) in enumerate(self.metrics.iterrows()):
                values = [float(row[name]) * 100 for name in metric_names]
                offset = (model_index - (len(model_names) - 1) / 2) * bar_width
                positions = [x + offset for x in x_positions]
                bars = ax.bar(
                    positions, values, width=bar_width,
                    label=str(row["Model"]),
                    color=model_colors[model_index % len(model_colors)],
                    edgecolor=C["surface"], linewidth=0.7
                )
                for bar, score in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.2,
                        f"{score:.1f}", ha="center", va="bottom",
                        fontsize=7, color=C["text"]
                    )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(metric_names, fontsize=8, color=C["muted"])
            ax.set_ylim(0, 112)
            ax.set_ylabel("Score (%)", fontsize=8, color=C["muted"])
            ax.tick_params(axis="y", labelsize=8, colors=C["muted"])
            ax.grid(axis="y", linestyle="--", alpha=0.18, color=C["primary"])
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color(C["border"])
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.17),
                ncol=2, frameon=False, fontsize=8
            )

        def risk_distribution_chart(ax):
            ax.clear()
            if not stats or total == 0:
                ax.text(
                    0.5, 0.5, "Risk distribution data not available",
                    ha="center", va="center", color=C["muted"], fontsize=9
                )
                ax.axis("off")
                return

            categories = ["High delivery risk", "Low delivery risk"]
            values = [high, low]
            colors = [C["orchid"], C["primary"]]
            bars = ax.barh(
                categories, values, color=colors, height=0.48,
                edgecolor=C["surface"], linewidth=0.8
            )
            ax.invert_yaxis()
            ax.set_xlim(0, max(values) * 1.22)
            ax.set_xlabel("Number of orders", fontsize=8, color=C["muted"])
            ax.tick_params(axis="y", labelsize=8, colors=C["text"], length=0)
            ax.tick_params(axis="x", labelsize=8, colors=C["muted"])
            ax.grid(axis="x", linestyle="--", alpha=0.18, color=C["primary"])
            ax.set_axisbelow(True)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(C["border"])
            for bar, count, pct in zip(bars, values, [high_pct, low_pct]):
                ax.text(
                    bar.get_width() + max(values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count:,}  ({pct:.1f}%)",
                    va="center", fontsize=8, color=C["text"]
                )

        model_card = self._add_chart(
            chart_row, "Model performance comparison",
            model_metrics_chart, height=3.0
        )
        model_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        risk_card = self._add_chart(
            chart_row, "Late-delivery risk distribution",
            risk_distribution_chart, height=3.0
        )
        risk_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        # A small project-status note keeps the dashboard understandable
        # without suggesting that missing model files are available.
        status_card = ctk.CTkFrame(
            self.content, fg_color=C["surface"], corner_radius=14,
            border_width=1, border_color=C["border"]
        )
        status_card.pack(fill="x", pady=(17, 8))
        ctk.CTkLabel(
            status_card, text="PROJECT STATUS",
            text_color=C["primary"],
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 4))

        model_status = (
            "Saved Random Forest model detected."
            if MODEL_PATH.exists()
            else "Prediction model file is not available yet. Prediction integration can be completed when the ML teammate shares the trained model."
        )
        ctk.CTkLabel(
            status_card, text=model_status,
            text_color=C["muted"], font=ctk.CTkFont(size=10),
            wraplength=900, justify="left"
        ).pack(anchor="w", padx=16, pady=(0, 12))

    # ------------------------- OTHER GUI SCREENS ----------------------------
    # These are intentionally simple placeholders for now.
    # We can develop each screen separately later without changing the dashboard.

    def _prediction_page(self):
        self._section_title(
            self.content,
            "Risk Prediction",
            "This screen will be developed in the next step."
        )
        notice = ctk.CTkFrame(
            self.content,
            fg_color=C["surface"],
            corner_radius=14,
            border_width=1,
            border_color=C["border"]
        )
        notice.pack(fill="x", pady=18)
        ctk.CTkLabel(
            notice,
            text="Risk Prediction GUI is coming next.",
            text_color=C["text"],
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=18, pady=(18, 6))
        ctk.CTkLabel(
            notice,
            text=(
                "The trained model file is not required for today's "
                "dashboard work. Prediction will be connected after "
                "the ML teammate shares the saved model."
            ),
            text_color=C["muted"],
            font=ctk.CTkFont(size=11),
            wraplength=700,
            justify="left"
        ).pack(anchor="w", padx=18, pady=(0, 18))

    def _what_if_page(self):
        self._section_title(
            self.content,
            "What-If Scenario Analysis",
            "This screen will be developed in the next step."
        )
        ctk.CTkLabel(
            self.content,
            text="What-If Analysis GUI is coming next.",
            text_color=C["muted"],
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=18)

    def _high_risk_page(self):
        self._section_title(
            self.content,
            "High-Risk Monitoring",
            "This screen will be developed in the next step."
        )
        ctk.CTkLabel(
            self.content,
            text="High-Risk Monitoring GUI is coming next.",
            text_color=C["muted"],
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=18)

    # ----------------------------- MODEL HELPERS ---------------------------
    def _load_model(self):
        if self.model is not None:
            return self.model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "The saved model 'models/random_forest_model.pkl' is not available. "
                "Ask the ML teammate to provide the already-trained model. "
                "This GUI will not train or recreate it."
            )
        if joblib is None:
            raise ImportError("joblib is required to load the saved model.")
        self.model = joblib.load(MODEL_PATH)
        if self.processed is None:
            raise FileNotFoundError("processed/supply_chain_processed.csv is missing.")
        self.feature_columns = [c for c in self.processed.columns if c != TARGET]
        return self.model

    def _prepare_model_input(self, order):
        self._load_model()
        frame = pd.DataFrame([order])
        encoded = pd.get_dummies(
            frame, columns=CAT_COLUMNS, drop_first=True, dtype="int8"
        )
        encoded = encoded.reindex(columns=self.feature_columns, fill_value=0)
        return encoded

    def _predict(self, order):
        model_input = self._prepare_model_input(order)
        prediction = int(self.model.predict(model_input)[0])
        probabilities = self.model.predict_proba(model_input)[0]
        classes = list(self.model.classes_)
        high_index = classes.index(1)
        return prediction, float(probabilities[high_index])

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

if __name__ == "__main__":
    app = SupplyChainApp()