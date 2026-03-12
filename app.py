import gradio as gr
import pandas as pd
from courier_delivery_time_estimator import process_and_predict

def predict_delivery_days(distance_in_km, from_pincode, to_latitude, to_longitude, to_state, courier_partner):
    if not all([distance_in_km, from_pincode, to_latitude, to_longitude, to_state, courier_partner]):
        return "⚠️ Please fill all fields."

    new_data = pd.DataFrame([{
        "distance_in_km": float(distance_in_km),
        "from_pincode": str(from_pincode),
        "to_latitude": float(to_latitude),
        "to_longitude": float(to_longitude),
        "to_state": to_state,
        "courier_partner": courier_partner
    }])

    result_df = process_and_predict(new_data)

    delivery_days = result_df.loc[0, "delivery_days"]

    # round to nearest 10
    rounded_days = round(delivery_days / 10) * 10

    return f"📦 Estimated Delivery Time: {rounded_days} days"

demo = gr.Interface(
fn=predict_delivery_days,
inputs=[
gr.Number(label="Distance (km)"),
gr.Textbox(label="From Pincode"),
gr.Number(label="To Latitude"),
gr.Number(label="To Longitude"),
gr.Textbox(label="Destination State"),
gr.Textbox(label="Courier Partner")
],
outputs="text",
title="🚚 Courier Delivery Time Estimator",
description="Enter shipment details to estimate delivery time."
)

if __name__ == "__main__":
    demo.launch()
