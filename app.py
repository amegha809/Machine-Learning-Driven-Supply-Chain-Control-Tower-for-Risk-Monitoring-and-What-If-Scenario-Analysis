
import customtkinter as ctk
import pandas as pd
import joblib

# Appearance settings
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Main application window
app = ctk.CTk()
app.title("Supply Chain Control Tower")
app.geometry("1100x700")
app.minsize(900, 600)

# Main layout
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

# Sidebar
sidebar = ctk.CTkFrame(app, width=220, corner_radius=0)
sidebar.grid(row=0, column=0, sticky="nsew")
sidebar.grid_propagate(False)

title = ctk.CTkLabel(
    sidebar,
    text="SUPPLY CHAIN\nCONTROL TOWER",
    font=ctk.CTkFont(size=19, weight="bold")
)
title.pack(padx=15, pady=(30, 35))


# Navigation buttons

def show_page(page_name):
    heading.configure(text=page_name)

    for widget in page_area.winfo_children():
        widget.destroy()

    if page_name == "Dashboard":
        subtitle.configure(
            text="Monitor delivery risks and supply chain performance"
        )
        for card in dashboard_cards:
            card.grid()

    else:
        for card in dashboard_cards:
            card.grid_remove()

        if page_name == "Risk Prediction":
            subtitle.configure(
                text="Enter order details to predict late delivery risk"
            )

            # Input fields
            fields = [
                ("Days for shipment (scheduled)", "3"),
                ("Benefit per order", "20"),
                ("Sales per customer", "150"),
                ("Order Item Discount", "10"),
                ("Order Item Discount Rate", "0.10"),
                ("Order Item Quantity", "1"),
                ("Sales", "150"),
                ("Order Item Total", "140"),
                ("Order Profit Per Order", "20"),
                ("Product Price", "150")
            ]

            entries = {}

            for i, (name, default) in enumerate(fields):
                row = i // 2
                col = (i % 2) * 2

                ctk.CTkLabel(
                    page_area,
                    text=name
                ).grid(
                    row=row,
                    column=col,
                    padx=8,
                    pady=8,
                    sticky="w"
                )

                entry = ctk.CTkEntry(
                    page_area,
                    width=160
                )
                entry.insert(0, default)
                entry.grid(
                    row=row,
                    column=col + 1,
                    padx=8,
                    pady=8
                )

                entries[name] = entry

            # Categorical inputs
            options = {
                "Type": ["DEBIT", "TRANSFER", "PAYMENT"],
                "Customer Segment": [
                    "Consumer", "Corporate", "Home Office"
                ],
                "Market": [
                    "Africa", "Europe", "LATAM",
                    "Pacific Asia", "USCA"
                ],
                "Shipping Mode": [
                    "Standard Class", "Second Class",
                    "First Class", "Same Day"
                ]
            }

            combos = {}
            start_row = 5

            for i, (name, values) in enumerate(options.items()):
                row = start_row + i

                ctk.CTkLabel(
                    page_area,
                    text=name
                ).grid(
                    row=row,
                    column=0,
                    padx=8,
                    pady=6,
                    sticky="w"
                )

                combo = ctk.CTkComboBox(
                    page_area,
                    values=values,
                    width=200
                )
                combo.set(values[0])
                combo.grid(
                    row=row,
                    column=1,
                    padx=8,
                    pady=6,
                    sticky="w"
                )

                combos[name] = combo

            result_label = ctk.CTkLabel(
                page_area,
                text="Prediction will appear here",
                font=ctk.CTkFont(size=16, weight="bold"),
                wraplength=650
            )
            result_label.grid(
                row=10,
                column=0,
                columnspan=4,
                padx=8,
                pady=15,
                sticky="w"
            )

            def predict_order():
                try:
                    order = {
                        name: float(entry.get())
                        for name, entry in entries.items()
                    }

                    order.update({
                        name: combo.get()
                        for name, combo in combos.items()
                    })

                    model = joblib.load(
                        "models/random_forest_model.pkl"
                    )

                    processed = pd.read_csv(
                        "processed/supply_chain_processed.csv",
                        nrows=1
                    )
                    feature_columns = [
                        col for col in processed.columns
                        if col != "Late_delivery_risk"
                    ]

                    order_df = pd.DataFrame([order])

                    encoded = pd.get_dummies(
                        order_df,
                        columns=[
                            "Type",
                            "Customer Segment",
                            "Market",
                            "Shipping Mode"
                        ],
                        drop_first=True,
                        dtype="int8"
                    )

                    encoded = encoded.reindex(
                        columns=feature_columns,
                        fill_value=0
                    )

                    prediction = model.predict(encoded)[0]
                    probabilities = model.predict_proba(encoded)[0]

                    risk = "HIGH" if prediction == 1 else "LOW"

                    result_label.configure(
                        text=(
                            f"Predicted Risk: {risk}\n"
                            f"Low Risk Probability: "
                            f"{probabilities[0] * 100:.2f}%\n"
                            f"High Risk Probability: "
                            f"{probabilities[1] * 100:.2f}%"
                        )
                    )

                except Exception as error:
                    result_label.configure(
                        text=f"Prediction error: {error}"
                    )

            predict_button = ctk.CTkButton(
                page_area,
                text="Predict Delivery Risk",
                command=predict_order
            )
            predict_button.grid(
                row=11,
                column=0,
                columnspan=2,
                padx=8,
                pady=10,
                sticky="w"
            )

        
        elif page_name == "What-If Analysis":
            subtitle.configure(
                text="Compare delivery risk across shipping modes"
            )

            ctk.CTkLabel(
                page_area,
                text="Shipping Scenario Analysis",
                font=ctk.CTkFont(size=20, weight="bold")
            ).pack(anchor="w", padx=10, pady=10)

            ctk.CTkLabel(
                page_area,
                text=(
                    "This analysis compares the model's predicted risk "
                    "for the same order using different shipping modes."
                ),
                wraplength=650
            ).pack(anchor="w", padx=10, pady=5)

            result_label = ctk.CTkLabel(
                page_area,
                text="Click the button to compare scenarios.",
                justify="left",
                font=ctk.CTkFont(size=15)
            )
            result_label.pack(anchor="w", padx=10, pady=15)

            def run_scenarios():
                try:
                    import subprocess
                    import sys

                    result = subprocess.run(
                        [
                            sys.executable,
                            "what_if_analysis.py"
                        ],
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    result_label.configure(
                        text=(
                            "Scenario analysis completed.\n\n"
                            + result.stdout[-2500:]
                        )
                    )

                except subprocess.CalledProcessError as error:
                    result_label.configure(
                        text="Scenario analysis failed:\n"
                        + error.stderr[-1500:]
                    )
                except Exception as error:
                    result_label.configure(
                        text=f"Error: {error}"
                    )

            ctk.CTkButton(
                page_area,
                text="Run What-If Analysis",
                command=run_scenarios
            ).pack(anchor="w", padx=10, pady=10)

        else:
            subtitle.configure(text=page_name)
            ctk.CTkLabel(
                page_area,
                text="This feature will be connected in the next step.",
                font=ctk.CTkFont(size=16)
            ).pack(padx=20, pady=30)


# Keep references to dashboard cards
dashboard_cards = []

for page in [
    "Dashboard",
    "Risk Prediction",
    "What-If Analysis",
    "High-Risk Orders"
]:
    button = ctk.CTkButton(
        sidebar,
        text=page,
        height=40,
        anchor="w",
        command=lambda name=page: show_page(name)
    )
    button.pack(fill="x", padx=15, pady=7)

# Main content area
content = ctk.CTkFrame(app, corner_radius=0, fg_color="transparent")
# Area for page-specific content
page_area = ctk.CTkFrame(content, fg_color="transparent")
page_area.grid(row=2, column=0, columnspan=2, sticky="nsew")
content.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
content.grid_columnconfigure((0, 1), weight=1)

heading = ctk.CTkLabel(
    content,
    text="Dashboard",
    font=ctk.CTkFont(size=28, weight="bold")
)
heading.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

subtitle = ctk.CTkLabel(
    content,
    text="Monitor delivery risks and supply chain performance",
    font=ctk.CTkFont(size=14),
    text_color="gray"
)
subtitle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 25))

# Dashboard cards
cards = [
    ("Project Status", "Phase 2 Development"),
    ("ML Models", "2 Models Trained"),
    ("Risk Analysis", "Available"),
    ("Scenario Analysis", "Available")
]

for index, (card_title, card_value) in enumerate(cards):
    card = ctk.CTkFrame(content, corner_radius=12)
    card.grid(
        row=2 + index // 2,
        column=index % 2,
        sticky="nsew",
        padx=8,
        pady=8
    )
    dashboard_cards.append(card)
    ctk.CTkLabel(
        card,
        text=card_title,
        font=ctk.CTkFont(size=16, weight="bold")
    ).pack(anchor="w", padx=18, pady=(20, 10))

    ctk.CTkLabel(
        card,
        text=card_value,
        font=ctk.CTkFont(size=20)
    ).pack(anchor="w", padx=18, pady=(0, 20))

# Footer
footer = ctk.CTkLabel(
    content,
    text="Machine Learning-Based Supply Chain Risk Monitoring",
    text_color="gray"
)
footer.grid(row=4, column=0, columnspan=2, pady=25)

# Start application
app.mainloop()
