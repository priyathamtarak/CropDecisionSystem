from flask import Flask, request, render_template
import numpy as np
import pickle

# importing model
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# creating flask app
app = Flask(__name__)

def get_suggested_crops_for_season_and_city(season, city):
    # Define a mapping between season, city, and suggested crops
    suggestions_map = {
        'summer': {
            'delhi': ['Cotton', 'Watermelon', 'Maize'],
            'kerala': ['Rice', 'Coconut', 'Banana'],
            'bangalore': ['Rice', 'Papaya', 'Drumstick'],
            'nellore': ['Cotton', 'Maize', 'Sunflower'],
            'hyderabad': ['Cotton', 'Groundnuts', 'Maize'],
            'secunderabad': ['Maize', 'Groundnuts', 'Sunflower'],
            'visakhapatnam': ['Maize', 'Cotton', 'Sunflower'],
            'guntur': ['Chillies', 'Tobacco', 'Cotton'],
            'krishna': ['Rice', 'Sugarcane', 'Mango'],
            'east godavari': ['Rice', 'Coconut', 'Banana'],
            'west godavari': ['Rice', 'Sugarcane', 'Mango'],
            'prakasam': ['Cotton', 'Maize', 'Groundnuts'],
            'kadapa': ['Maize', 'Pulses', 'Sunflower'],

            # Add more city-specific suggestions for summer if needed
        },
        'winter': {
            'delhi': ['Wheat', 'Mustard', 'Tomato'],
            'mumbai': ['Rice', 'Sugarcane', 'Banana'],
            'kerala': ['Wheat', 'Kidneybean', 'Pomegranate'],
            'bangalore': ['Rice', 'Papaya', 'Drumstick'],
            'nellore': ['Wheat', 'Barley', 'Mustard'],
            'hyderabad': ['Chickpeas', 'Wheat', 'Barley'],
            'secunderabad': ['Chickpeas', 'Barley', 'Mustard'],
            'visakhapatnam': ['Barley', 'Mustard', 'Wheat'],
            'guntur': ['Wheat', 'Barley', 'Pulses'],
            'krishna': ['Wheat', 'Barley', 'Groundnuts'],
            'east godavari': ['Wheat', 'Barley', 'Sesame'],
            'west godavari': ['Wheat', 'Barley', 'Chillies'],
            'prakasam': ['Wheat', 'Barley', 'Pulses'],
            'kadapa': ['Wheat', 'Barley', 'Mustard'],

            # Add more city-specific suggestions for winter if needed
        },
        'rainy': {
            'nellore': ['Rice', 'Maize', 'Pulses'],
            'hyderabad': ['Okra', 'Brinjal', 'Tomato'],
            'secunderabad': ['Rice', 'Brinjal', 'Okra'],
            'visakhapatnam': ['Maize', 'Pulses', 'Tomato'],
            'guntur': ['Rice', 'Pulses', 'Chillies'],
            'krishna': ['Rice', 'Maize', 'Pulses'],
            'east godavari': ['Rice', 'Maize', 'Pulses'],
            'west godavari': ['Rice', 'Maize', 'Pulses'],
            'prakasam': ['Rice', 'Maize', 'Pulses'],
            'kadapa': ['Rice', 'Maize', 'Pulses'],
        }
        # Add more seasons and city combinations as necessary
    }

    # Check if the given season and city combination exists in the mapping
    if season.lower() in suggestions_map:
        if city.lower() in suggestions_map[season.lower()]:
            return suggestions_map[season.lower()][city.lower()]
        else:
            return ['No specific suggestions for the given city in this season']
    else:
        return ['No specific suggestions for the given season']

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Extracting input values
    N = request.form.get('Nitrogen')
    P = request.form.get('Phosporus')
    K = request.form.get('Potassium')
    temp = request.form.get('Temperature')
    humidity = request.form.get('Humidity')
    ph = request.form.get('Ph')
    rainfall = request.form.get('Rainfall')
    season = request.form.get('Season')
    city = request.form.get('City')

    # Creating a feature list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Transforming features using MinMaxScaler
    scaled_features = ms.transform(single_pred)
    # Transforming the scaled features using StandardScaler
    final_features = sc.transform(scaled_features)

    # Making predictions using the model
    prediction = model.predict(final_features)

    # Mapping prediction to crop names
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Building the result message for crop recommendation based on ML prediction
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result_crop = f"{crop} is the best crop to be cultivated right there"
    else:
        result_crop = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Getting suggested crops based on season and city
    suggested_crops = get_suggested_crops_for_season_and_city(season, city)

    # Consolidating the result message
    result_message = f"{result_crop}. "
    result_message += f"Suggested crops for {season.lower()} season in {city.lower()} city: {', '.join(suggested_crops)}."

    print("Received form data:", feature_list)
    print("Predicted crop:", crop)
    print("Suggested crops for season and city:", suggested_crops)
    print("Result: ",result_message)

    return render_template('index.html', result=result_message)

# python main
if __name__ == "__main__":
    app.run(debug=True)
