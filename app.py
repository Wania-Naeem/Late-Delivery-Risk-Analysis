from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import shap
import json

app = Flask(__name__)
CORS(app)

# ---- Load model + supporting files (same as before) ----
model = joblib.load('late_delivery_model.pkl')
model_columns = joblib.load('model_columns.pkl')
dropdown_options = joblib.load('dropdown_options.pkl')
explainer = shap.TreeExplainer(model)

country_region_map = joblib.load('country_region_map.pkl')
country_market_map = joblib.load('country_market_map.pkl')
country_states_map = joblib.load('country_states_map.pkl')

# ---- Load precomputed dashboard data (dataset stats, model metrics, EDA, importances) ----
with open('precomputed_data.json') as f:
    PRECOMPUTED = json.load(f)


def prettify_feature_name(feature_name):
    friendly_map = {
        'Days for shipment (scheduled)': 'Scheduled shipping days',
        'Order Item Quantity': 'Order quantity',
        'Sales': 'Sales amount',
        'Order Item Discount Rate': 'Discount rate',
        'order_month': 'Order month',
        'order_day_of_week': 'Day of week',
        'order_quarter': 'Order quarter',
        'Product Price': 'Product price',
        'Benefit per order': 'Profit per order'
    }
    if feature_name in friendly_map:
        return friendly_map[feature_name]
    for prefix in ['Shipping Mode_', 'Order Region_', 'Market_', 'Category Name_',
                   'Customer Segment_', 'Type_', 'Order Country_', 'Order State_']:
        if feature_name.startswith(prefix):
            category = prefix.rstrip('_')
            value = feature_name[len(prefix):]
            return f"{category}: {value}"
    return feature_name


def generate_suggestions(top_factors):
    suggestions = []
    for factor_name, impact in top_factors:
        if impact <= 0:
            continue
        if 'Shipping Mode: First Class' in factor_name or 'Shipping Mode: Same Day' in factor_name:
            suggestions.append("Consider switching to Standard or Second Class shipping — faster shipping modes counter-intuitively show higher delay rates in this data, likely due to tighter scheduling margins.")
        elif 'Scheduled shipping days' in factor_name:
            suggestions.append("Consider allocating more scheduled shipping days for this route — tight schedules increase the risk of missing the delivery window.")
        elif 'Discount rate' in factor_name:
            suggestions.append("High discount rates are associated with higher delay risk in this data — review if heavily discounted orders get deprioritized in fulfillment.")
        elif 'Order quantity' in factor_name:
            suggestions.append("Large order quantities may require more processing time — consider splitting very large orders into multiple shipments.")
        elif 'Order Region' in factor_name or 'Order Country' in factor_name or 'Order State' in factor_name:
            suggestions.append(f"This destination ({factor_name.split(': ')[-1]}) has historically higher delay rates — consider reviewing the logistics partner or route for this region.")
        elif 'Order month' in factor_name or 'Order quarter' in factor_name:
            suggestions.append("This time period has historically shown higher delay rates — possibly due to seasonal demand spikes. Consider building in extra buffer time.")
        elif 'Day of week' in factor_name:
            suggestions.append("Orders placed on this day of the week show a higher historical delay rate — consider adjusting order timing or building in extra buffer if possible.")
        elif 'Product price' in factor_name:
            suggestions.append("Higher-priced items in this data show a higher historical delay rate — consider double-checking fulfillment priority for high-value items.")
        elif 'Sales amount' in factor_name:
            suggestions.append("Larger order sales values are associated with higher delay risk in this data — consider verifying stock/handling capacity for higher-value orders.")
        elif 'Profit per order' in factor_name:
            suggestions.append("This order's profit margin profile is associated with higher delay risk historically — consider reviewing fulfillment prioritization rules tied to margin.")
        else:
            suggestions.append(f"'{factor_name}' is increasing this order's delay risk based on historical patterns — consider reviewing this factor for this order.")

    if not suggestions:
        suggestions.append("This order's risk factors are balanced — no single dominant cause identified. Overall monitoring is recommended.")

    return suggestions[:3]


# ---------------- Dashboard pages ----------------

@app.route('/')
def dashboard():
    return render_template('dashboard.html', page_title='Dashboard', active='dashboard',
                            overview=PRECOMPUTED['dataset_overview'],
                            final_model=PRECOMPUTED['final_model'])


@app.route('/data-analysis')
def data_analysis():
    return render_template('data_analysis.html', page_title='Data analysis', active='data_analysis',
                            eda=PRECOMPUTED['eda'])


@app.route('/model-performance')
def model_performance():
    return render_template('model_performance.html', page_title='Model performance', active='model_performance',
                            model_comparison=PRECOMPUTED['model_comparison'],
                            final_model=PRECOMPUTED['final_model'])


@app.route('/explainability')
def explainability():
    return render_template('explainability.html', page_title='Explainability', active='explainability',
                            feature_importance=PRECOMPUTED['feature_importance'])


@app.route('/prediction')
def prediction():
    return render_template('prediction.html', page_title='Prediction', active='prediction')


@app.route('/dataset')
def dataset_page():
    return render_template('dataset.html', page_title='Dataset', active='dataset',
                            overview=PRECOMPUTED['dataset_overview'],
                            dataset_columns=PRECOMPUTED['dataset_columns'])


# ---------------- Existing prediction API (unchanged logic) ----------------

@app.route('/dropdown-options', methods=['GET'])
def get_dropdown_options():
    return jsonify(dropdown_options)


@app.route('/country-mapping', methods=['GET'])
def get_country_mapping():
    return jsonify({
        'region_map': country_region_map,
        'market_map': country_market_map,
        'states_map': country_states_map
    })


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    input_dict = {
        'Days for shipment (scheduled)': float(data['days_scheduled']),
        'Order Item Quantity': float(data['order_qty']),
        'Sales': float(data['sales']),
        'Order Item Discount Rate': float(data['discount_rate']),
        'order_month': int(data['order_month']),
        'order_day_of_week': int(data['order_day_of_week']),
        'order_quarter': int(data['order_quarter']),
        'Product Price': float(data['product_price']),
        'Benefit per order': float(data['benefit_per_order']),
        'Shipping Mode': data['shipping_mode'],
        'Order Region': data['order_region'],
        'Market': data['market'],
        'Category Name': data['category_name'],
        'Customer Segment': data['customer_segment'],
        'Type': data['order_type'],
        'Order Country': data['order_country'],
        'Order State': data['order_state']
    }

    input_df = pd.DataFrame([input_dict])

    categorical_cols = ['Shipping Mode', 'Order Region', 'Market', 'Category Name',
                         'Customer Segment', 'Type', 'Order Country', 'Order State']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction_result = int(model.predict(input_final)[0])
    probability = float(model.predict_proba(input_final)[0][1])

    shap_values = explainer.shap_values(input_final, check_additivity=False)
    if isinstance(shap_values, list):
        class1_shap = shap_values[1][0]
    else:
        class1_shap = shap_values[0]

    categorical_prefixes = ['Shipping Mode_', 'Order Region_', 'Market_', 'Category Name_',
                             'Customer Segment_', 'Type_', 'Order Country_', 'Order State_']

    def is_relevant_feature(feat_name, value):
        is_categorical_dummy = any(feat_name.startswith(p) for p in categorical_prefixes)
        if not is_categorical_dummy:
            return True
        return value == 1

    feature_impacts = list(zip(input_final.columns, class1_shap, input_final.iloc[0].values))
    feature_impacts = [(f, i) for f, i, v in feature_impacts if is_relevant_feature(f, v)]
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    top_factors = []
    for feat, impact in feature_impacts[:5]:
        if abs(impact) > 0.001:
            friendly_name = prettify_feature_name(feat)
            direction = "increased" if impact > 0 else "decreased"
            top_factors.append({
                'feature': friendly_name,
                'direction': direction,
                'impact': round(float(impact), 4)
            })

    suggestions = generate_suggestions([(f['feature'], f['impact']) for f in top_factors])

    return jsonify({
        'prediction': prediction_result,
        'probability': round(probability * 100, 1),
        'top_factors': top_factors,
        'suggestions': suggestions
    })


if __name__ == '__main__':
    app.run(debug=False, port=5000, use_reloader=False)
