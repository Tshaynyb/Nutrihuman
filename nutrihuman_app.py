import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import sqlite3
from datetime import datetime
import os
import uuid

# Load a pre-trained model (Note: This is ImageNet weights. For food-specific, consider a Food-101 model if available)
model = tf.keras.applications.EfficientNetB0(weights='imagenet') # Replace with Food-101 if available


# Function to analyze meal images
def analyze_meal(image_path):
try:
# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict ingredients using the model
predictions = model.predict(img_array)
top_preds = decode_predictions(predictions, top=5)[0]

# Extract food-related items from predictions
detected_ingredients = [pred[1] for pred in top_preds] # pred[1] contains label
return detected_ingredients
except Exception as e:
st.error(f"Error analyzing meal: {e}")
return []


# User profile class
class UserProfile:
def __init__(self, name, dob, height, weight):
self.name = name
self.dob = dob
self.height = height # in inches
self.weight = weight # in pounds
self.age = self.calculate_age()
self.bmi = self.calculate_bmi()
self.goal = None
self.health_categories = []

def calculate_age(self):
try:
today = datetime.today()
birth_date = datetime.strptime(self.dob, "%m/%d/%Y")
age = today.year - birth_date.year
if (today.month, today.day) < (birth_date.month, birth_date.day):
age -= 1
return age
except Exception as e:
st.error(f"Error calculating age: {e}")
return 0

def calculate_bmi(self):
try:
height_meters = self.height * 0.0254 # Inches to meters
weight_kg = self.weight * 0.453592 # Pounds to kg
bmi = weight_kg / (height_meters ** 2)
return round(bmi, 1)
except Exception as e:
st.error(f"Error calculating BMI: {e}")
return 0

def set_goal(self, goal):
self.goal = goal

def add_health_category(self, category):
if category not in self.health_categories:
self.health_categories.append(category)

def daily_calorie_needs(self):
try:
# Using Mifflin-St Jeor Equation
weight_kg = self.weight * 0.453592
height_cm = self.height * 2.54
bmr = 10 * weight_kg + 6.25 * height_cm - 5 * self.age + 5
activity_factor = 1.2 # Sedentary as default
if self.goal == "lose_weight":
return int(bmr * activity_factor - 500)
elif self.goal == "gain_weight":
return int(bmr * activity_factor + 500)
else:
return int(bmr * activity_factor) # Maintenance
except Exception as e:
st.error(f"Error calculating calorie needs: {e}")
return 2000


# Meal and juice plans
class MealPlans:
def __init__(self, user_profile):
self.user_profile = user_profile

def generate_weekly_meal_plan(self):
meal_plan = {}
categories = self.user_profile.health_categories
for category in categories:
if category == "performance_energy":
meal_plan.update({
"Monday": "Quinoa bowl with hemp seeds, walnuts, and kale.",
"Tuesday": "Chickpea salad with dandelion greens and watercress.",
"Wednesday": "Amaranth porridge with banana and berries.",
"Thursday": "Fonio stir-fry with okra and soursop juice.",
"Friday": "Sea moss smoothie with dates and figs.",
"Saturday": "Walnut milk with banana and chia seeds.",
"Sunday": "Grilled tofu with kale and quinoa."
})
elif category == "digestive_health":
meal_plan.update({
"Monday": "Teff porridge with figs and tamarind juice.",
"Tuesday": "Wild rice salad with nopales and okra.",
"Wednesday": "Chickpea curry with dandelion greens.",
"Thursday": "Sea moss smoothie with papaya and aloe vera water.",
"Friday": "Fonio with steamed vegetables and tamarind juice.",
"Saturday": "Fermented vegetables with hemp seeds.",
"Sunday": "Quinoa bowl with kale and figs."
})
# Add more categories as needed...
return meal_plan

def generate_weekly_juice_plan(self):
juice_plan = {}
categories = self.user_profile.health_categories
for category in categories:
if category == "performance_energy":
juice_plan.update({
"Monday": "Banana smoothie.",
"Tuesday": "Soursop juice.",
"Wednesday": "Walnut milk.",
"Thursday": "Date and fig juice.",
"Friday": "Berry blast smoothie.",
"Saturday": "Green juice with kale and cucumber.",
"Sunday": "Hemp seed milk."
})
elif category == "digestive_health":
juice_plan.update({
"Monday": "Papaya and tamarind juice.",
"Tuesday": "Aloe vera water.",
"Wednesday": "Sea moss smoothie.",
"Thursday": "Ginger and tamarind shot.",
"Friday": "Cucumber and mint juice.",
"Saturday": "Apple and celery juice.",
"Sunday": "Kale and pear juice."
})
# Add more categories as needed...
return juice_plan

# Initialize database
def init_db():
conn = sqlite3.connect('nutrihuman.db', check_same_thread=False)
cursor = conn.cursor()

# Create tables for users and meal plans
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT NOT NULL,
dob TEXT NOT NULL,
height REAL NOT NULL,
weight REAL NOT NULL,
goal TEXT NOT NULL,
health_categories TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS meal_plans (
id INTEGER PRIMARY KEY AUTOINCREMENT,
user_id INTEGER NOT NULL,
day TEXT NOT NULL,
meal TEXT NOT NULL,
FOREIGN KEY (user_id) REFERENCES users (id)
)
''')

conn.commit()
return conn

# Add user to database
def add_user(conn, name, dob, height, weight, goal, health_categories):
cursor = conn.cursor()
cursor.execute('''
INSERT INTO users (name, dob, height, weight, goal, health_categories)
VALUES (?, ?, ?, ?, ?, ?)
''', (name, dob, height, weight, goal, ",".join(health_categories)))
conn.commit()
return cursor.lastrowid

# Add meal plan to database
def add_meal_plan(conn, user_id, day, meal):
cursor = conn.cursor()
cursor.execute('''
INSERT INTO meal_plans (user_id, day, meal)
VALUES (?, ?, ?)
''', (user_id, day, meal))
conn.commit()

# Retrieve meal plan
def get_meal_plan(conn, user_id):
cursor = conn.cursor()
cursor.execute('''
SELECT day, meal FROM meal_plans WHERE user_id=?
''', (user_id,))
return cursor.fetchall()


# Main Streamlit app
def main():
st.title("NutriHuman - Food & Wellness App")
st.write("Upload a meal image to analyze ingredients, generate meal & juice plans, post automatically with hashtags!")

# Initialize database
conn = init_db()

# Session state for user
if "user_id" not in st.session_state:
st.session_state.user_id = None

# Sidebar for profile
st.sidebar.header("User Profile")
name = st.sidebar.text_input("Name")
dob = st.sidebar.text_input("DOB (MM/DD/YYYY)")
height = st.sidebar.number_input("Height (in inches)", min_value=0.0)
weight = st.sidebar.number_input("Weight (pounds)", min_value=0.0)
goal = st.sidebar.selectbox("Goal", ["lose_weight", "gain_weight", "improve_energy", "support_digestion"])
health_categories = st.sidebar.multiselect("Categories", ["performance_energy", "digestive_health"])

if st.sidebar.button("Create Profile"):
if name and dob and height > 0 and weight > 0:
user_profile = UserProfile(name, dob, height, weight)
user_profile.set_goal(goal)
for cat in health_categories:
user_profile.add_health_category(cat)
user_id = add_user(conn, name, dob, height, weight, goal, health_categories)
st.session_state.user_id = user_id
st.sidebar.success("Profile created!")
else:
st.sidebar.error("Fill all fields correctly.")

if st.session_state.user_id:
# Load user profile (optional optimization)
pass

# Image upload
uploaded_file = st.file_uploader("Upload Meal Image", type=["jpg", "jpeg", "png"])
image_path = None
if uploaded_file:
os.makedirs("uploads", exist_ok=True)
unique_name = str(uuid.uuid4()) + ".jpg"
image_path = os.path.join("uploads", unique_name)
with open(image_path, "wb") as f:
f.write(uploaded_file.getbuffer())
img = Image.open(image_path)
st.image(img, caption="Uploaded Meal", use_column_width=True)
st.write("Analyzing...")
ingredients = analyze_meal(image_path)
st.write("Detected ingredients:")
st.write(ingredients)

# Generate and show plans
if st.button("Generate Weekly Plans"):
if st.session_state.user_id:
# Retrieve user data
# (Optional: cache or reload profile info from DB)
cursor = conn.cursor()
cursor.execute("SELECT name, dob, height, weight, goal, health_categories FROM users WHERE id=?", (st.session_state.user_id,))
user_data = cursor.fetchone()
if user_data:
name, dob, height, weight, goal, categories_str = user_data
categories = categories_str.split(",") if categories_str else []
user_profile = UserProfile(name, dob, height, weight)
user_profile.set_goal(goal)
for c in categories:
user_profile.add_health_category(c)
# Generate plans
meal_obj = MealPlans(user_profile)
weekly_meal_plan = meal_obj.generate_weekly_meal_plan()
weekly_juice_plan = meal_obj.generate_weekly_juice_plan()

st.subheader("Meal Plan")
for day, meal in weekly_meal_plan.items():
st.write(f"{day}: {meal}")

st.subheader("Juice Plan")
for day, juice in weekly_juice_plan.items():
st.write(f"{day}: {juice}")
else:
st.error("User profile not found. Please create profile.")
else:
st.error("Please create a profile first.")

# Run app
if __name__ == "__main__":
main()

