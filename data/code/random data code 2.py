import random
import pandas as pd

# -----------------------------
# Restaurants Data
# -----------------------------
restaurants = {
    "McDonalds": {
        "main": ["Big Mac", "McChicken", "Quarter Pounder"],
        "side": ["Fries", "Nuggets"],
        "drink": ["Coke", "Fanta", "Sprite"]
    },
    "KFC": {
        "main": ["Zinger", "Twister", "Chicken Bucket"],
        "side": ["Fries", "Coleslaw"],
        "drink": ["Pepsi", "7UP"]
    },
    "PizzaHut": {
        "main": ["Pepperoni Pizza", "Margherita", "Chicken Ranch"],
        "side": ["Wings", "Garlic Bread"],
        "drink": ["Pepsi", "Mirinda"]
    },
    "Dominos": {
        "main": ["Pepperoni Pizza", "BBQ Chicken Pizza"],
        "side": ["Potato Wedges", "Cheesy Bread"],
        "drink": ["Coke", "Sprite"]
    },
    "Bazooka": {
        "main": ["Crispy Sandwich", "Beef Burger"],
        "side": ["Fries", "Mozzarella Sticks"],
        "drink": ["Cola", "Juice"]
    },
    "CookDoor": {
        "main": ["Chicken Sandwich", "Shawarma"],
        "side": ["Fries", "Onion Rings"],
        "drink": ["Cola", "Juice"]
    },
    "Hardees": {
        "main": ["Thickburger", "Chicken Fillet"],
        "side": ["Fries", "Nuggets"],
        "drink": ["Pepsi", "Cola"]
    },
    "TacoBell": {
        "main": ["Taco", "Burrito"],
        "side": ["Nachos", "Fries"],
        "drink": ["Pepsi", "Mountain Dew"]
    },
    "SpaghettiFactory": {
        "main": ["Spaghetti Bolognese", "Alfredo Pasta"],
        "side": ["Garlic Bread", "Salad"],
        "drink": ["Juice", "Soda"]
    },
    "Zooba": {
        "main": ["Koshary", "Falafel Sandwich"],
        "side": ["Fries", "Pickles"],
        "drink": ["Tamarind Juice", "Hibiscus"]
    }
}

# -----------------------------
# Combos
# -----------------------------
combos = {
    "McDonalds": [["Big Mac", "Fries", "Coke"]],
    "KFC": [["Zinger", "Fries", "Pepsi"]],
    "PizzaHut": [["Pepperoni Pizza", "Wings", "Pepsi"]],
    "CookDoor": [["Shawarma", "Fries", "Juice"]],
    "Zooba": [["Koshary", "Tamarind Juice"]]
}

# -----------------------------
# Restaurant Pricing Factor
# -----------------------------
restaurant_factor = {
    "McDonalds": 1.2,
    "KFC": 1.2,
    "PizzaHut": 1.3,
    "Dominos": 1.3,
    "Bazooka": 1.0,
    "CookDoor": 0.9,
    "Hardees": 1.3,
    "TacoBell": 1.1,
    "SpaghettiFactory": 1.2,
    "Zooba": 1.0
}

# -----------------------------
# Price Ranges
# -----------------------------
price_ranges = {
    "main": (80, 150),
    "side": (30, 70),
    "drink": (20, 40)
}

# -----------------------------
# Assign Prices
# -----------------------------
def assign_prices(restaurants):
    prices = {}
    for name, menu in restaurants.items():
        factor = restaurant_factor.get(name, 1)

        for item in menu["main"]:
            prices[item] = int(random.randint(*price_ranges["main"]) * factor)

        for item in menu["side"]:
            prices[item] = int(random.randint(*price_ranges["side"]) * factor)

        for item in menu["drink"]:
            prices[item] = int(random.randint(*price_ranges["drink"]) * factor)

    return prices

prices = assign_prices(restaurants)
popularity = {item: 1 for item in prices}

# -----------------------------
# Helper Functions
# -----------------------------
def choose_by_price_and_popularity(items):
    weights = []
    for item in items:
        weight = (1 / prices[item]) * popularity[item]
        weights.append(weight)
    return random.choices(items, weights=weights)[0]

def update_popularity(items):
    for item in items:
        popularity[item] += 0.1

# -----------------------------
# User & Time
# -----------------------------
user_types = ["student", "family", "diet"]
time_slots = ["morning", "afternoon", "night"]

# -----------------------------
# Generate Order
# -----------------------------
def generate_order(order_id):
    restaurant = random.choice(list(restaurants.keys()))
    menu = restaurants[restaurant]

    user = random.choice(user_types)
    time = random.choice(time_slots)

    items = []

    # Combo logic
    if random.random() < 0.25 and restaurant in combos and user != "diet":
        items = random.choice(combos[restaurant])
        main_item = items[0]
    else:
        main_item = choose_by_price_and_popularity(menu["main"])
        items.append(main_item)

        # Side logic
        if user == "family":
            if random.random() < 0.9:
                items.append(choose_by_price_and_popularity(menu["side"]))
                if random.random() < 0.5:
                    items.append(choose_by_price_and_popularity(menu["side"]))
        elif user == "diet":
            if random.random() < 0.2:
                items.append(choose_by_price_and_popularity(menu["side"]))
        else:
            if random.random() < 0.7:
                items.append(choose_by_price_and_popularity(menu["side"]))

        # Drink
        if user != "diet" and random.random() < 0.8:
            items.append(choose_by_price_and_popularity(menu["drink"]))

    # Time effect
    if time == "night" and "Pizza" in main_item:
        items.append(choose_by_price_and_popularity(menu["drink"]))

    # Noise
    if random.random() < 0.05:
        items.append(random.choice(list(prices.keys())))

    # Update popularity
    update_popularity(items)

    return {
        "Order_ID": order_id,
        "Restaurant": restaurant,
        "User_Type": user,
        "Time": time,
        "Items_List": items,
        "Items": ", ".join(items)
    }

# -----------------------------
# Generate Dataset
# -----------------------------
orders = [generate_order(i) for i in range(1, 20001)]
df = pd.DataFrame(orders)

df.to_csv("final_dataset2.csv", index=False)

print(df.head())