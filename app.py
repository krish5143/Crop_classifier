# app.py
# Streamlit crop classifier UI with descriptions for 140 crops.
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import joblib

# ---------------------------
# Load trained model data (weights + class_to_idx)
# ---------------------------
model_data = joblib.load("crop_classifier_model.pkl")  # ensure file exists

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(model_data["class_to_idx"]))
model.load_state_dict(model_data["model_state_dict"])
model.eval()

# Reverse mapping index → class name
idx_to_class = {v: k for k, v in model_data["class_to_idx"].items()}

# ---------------------------
# Preprocessing pipeline
# ---------------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ---------------------------
# Plant descriptions for UI (2-3 lines each)
# Keys match your train dataset class names exactly.
# ---------------------------
plant_descriptions = {
    "Aji pepper plant": "A hot chili pepper native to South America, especially Peru. \
It adds heat and fruity flavor to traditional dishes like ceviche. \
Rich in vitamin C and capsaicin, boosting immunity and metabolism. \
Aji peppers are a key ingredient in Peruvian cuisine and sauces.",
    "Almonds plant": "A nut tree native to the Middle East and South Asia, now widely grown in California. \
The edible seeds are packed with protein, vitamin E, and healthy fats. \
Used in snacks, sweets, almond milk, and oils. \
Almond blossoms also support honey production by bees.",
    "Amaranth plant": "An ancient grain-like crop native to Central and South America. \
Both seeds and leaves are edible, offering high protein and minerals. \
Amaranth is gluten-free and used in porridges, breads, and curries. \
The Aztecs once considered it a sacred food crop.",
    "Apples plant": "One of the most widely grown fruits, native to Central Asia. \
Apples are rich in fiber, vitamin C, and antioxidants. \
Consumed fresh, juiced, or baked in desserts worldwide. \
China is currently the largest producer of apples.",
    "Artichoke plant": "A Mediterranean crop cultivated for its edible flower buds. \
Rich in fiber, vitamin C, and antioxidants supporting liver health. \
Consumed steamed, grilled, or used in dips like artichoke-spinach. \
Artichokes were prized in ancient Greek and Roman cuisine.",
    "Avocados plant": "Native to Central America and Mexico, thriving in tropical climates. \
Avocados are creamy fruits packed with healthy fats and potassium. \
Widely used in guacamole, spreads, and health smoothies. \
Nicknamed 'alligator pear' due to their rough green skin.",
    "Açaí plant": "A palm native to the Amazon rainforest in Brazil. \
Produces small purple berries rich in antioxidants and omega fatty acids. \
Often blended into juices, bowls, and smoothies. \
Açaí became a global health food trend in the 2000s.",
    "Bananas plant": "Native to Southeast Asia, now grown across tropical regions. \
Bananas are rich in potassium and carbohydrates for quick energy. \
Consumed fresh, baked, or fried, and used in desserts and smoothies. \
India is the world’s largest producer of bananas.",
    "Barley plant": "An ancient cereal grain domesticated in the Fertile Crescent. \
Barley is high in fiber and minerals, supporting digestive health. \
Used in soups, breads, and as malt in beer brewing. \
It was one of the first grains cultivated by humans.",
    "Beets plant": "A root vegetable originating in the Mediterranean region. \
Beets are rich in nitrates, iron, and antioxidants for heart health. \
Consumed roasted, juiced, or in salads and soups. \
Beet juice is used by athletes to boost stamina and blood flow.",
    "Black pepper plant": "A tropical vine native to India, especially Kerala. \
Produces peppercorns, the most traded spice in the world. \
Used for seasoning, preservation, and even traditional medicine. \
Known as 'black gold,' pepper was once worth its weight in silver.",
    "Blueberries plant": "Native to North America, now cultivated globally in cooler climates. \
Blueberries are rich in antioxidants, vitamin C, and fiber. \
Consumed fresh, in juices, jams, and baked goods. \
They are often called a 'superfood' for brain and heart health.",
    "Bok choy plant": "A leafy green vegetable from China, part of the cabbage family. \
Low in calories but rich in vitamins A, C, and K. \
Used in stir-fries, soups, and steamed dishes. \
Bok choy is a staple in East Asian cuisine and traditional medicine.",
    "Brazil nuts plant": "A massive tree native to the Amazon rainforest. \
Produces nuts extremely rich in selenium, vital for thyroid function. \
Consumed raw, roasted, or used in chocolates and desserts. \
Harvesting Brazil nuts helps preserve Amazonian forests.",
    "Broccoli plant": "A green vegetable in the cabbage family, native to the Mediterranean. \
Rich in vitamin C, fiber, and cancer-fighting compounds. \
Consumed steamed, roasted, or raw in salads. \
It became globally popular only in the 20th century.",
    "Brussels sprout plant": "Native to Belgium and popular across Europe and America. \
Small cabbage-like buds high in vitamin K, vitamin C, and fiber. \
Usually roasted, sautéed, or steamed as a side dish. \
They grow best in cool climates and improve in flavor after frost.",
    "Buckwheat plant": "A gluten-free pseudocereal native to Asia. \
Rich in protein, fiber, and essential minerals like magnesium. \
Used to make soba noodles, pancakes, and porridges. \
Despite the name, it is unrelated to wheat and safe for celiacs.",
    "Cabbages and other brassicas plant": "Cultivated globally in temperate regions. \
Cabbage is high in vitamin C, vitamin K, and dietary fiber. \
Used in salads, soups, curries, and pickling (sauerkraut, kimchi). \
It is one of the oldest cultivated leafy vegetables.",
    "Camucamu plant": "A shrub from the Amazon rainforest. \
Produces small sour berries with extremely high vitamin C levels. \
Used in juices, jams, and health supplements. \
It is considered one of the world’s best natural immune boosters.",
    "Carrots and turnips plant": "Root vegetables grown worldwide in temperate climates. \
Carrots are rich in beta-carotene for eye health. \
Turnips are high in vitamin C and used in soups and stews. \
Both are staple winter crops with long storage life.",
    "Cashew nuts plant": "A tropical tree native to northeastern Brazil. \
The cashew nut is a kidney-shaped seed, rich in healthy fats, protein, and copper. \
Used in snacks, nut butters, and dairy-free products. \
The cashew 'apple' is also edible, but often discarded.",
    "Cauliflower plant": "A vegetable in the cabbage family, native to the Mediterranean. \
It is rich in vitamin C, vitamin K, and antioxidants. \
Used in curries, roasted dishes, and as a low-carb rice or pizza crust alternative. \
Its name comes from the Latin 'caulis' (cabbage) and 'floris' (flower).",
    "Celery plant": "A marshland plant cultivated worldwide for its long fibrous stalks. \
It is low in calories, high in water content, and a good source of vitamin K. \
Used in salads, soups, and as a flavor base for many dishes. \
Celery seeds are also used as a spice and in herbal medicine.",
    "Cherimoya plant": "A fruit tree native to the Andes mountains of South America. \
The fruit has a creamy texture and a flavor often described as a mix of banana, pineapple, and strawberry. \
Rich in fiber, vitamin C, and antioxidants. \
Mark Twain called it 'the most delicious fruit known to men'.",
    "Cherry plant": "A fruit tree cultivated globally, originating from Europe and West Asia. \
Cherries are rich in antioxidants and anti-inflammatory compounds. \
Eaten fresh, used in pies, tarts, and jams, or processed into juice. \
They are a 'drupe' fruit, meaning they have a single hard seed in the center.",
    "Chestnuts plant": "A nut tree native to the northern hemisphere, particularly Europe, Asia, and North America. \
Chestnuts are a starchy nut, low in fat and high in vitamin C. \
They are roasted, boiled, or ground into flour for baking. \
Unlike most nuts, they are a primary source of carbohydrates.",
    "Chickpeas plant": "A legume cultivated in the Middle East and now globally, also known as garbanzo beans. \
High in protein, fiber, and iron, making them a staple in vegetarian diets. \
Used to make hummus, falafel, and added to salads and curries. \
They are a key part of Mediterranean, Middle Eastern, and Indian cuisines.",
    "Chili peppers and green peppers plant": "Cultivated varieties of the Capsicum genus, originating in the Americas. \
Peppers are rich in vitamin C and antioxidants. \
Used globally for their flavor and heat in a wide range of cuisines. \
The Scoville scale measures their heat based on capsaicin content.",
    "Cinnamon plant": "An evergreen tree native to Sri Lanka. \
The spice is made from the tree's inner bark, which curls into quills. \
Used globally as a flavoring agent in sweet and savory dishes, and for its medicinal properties. \
Cinnamon is known for its anti-inflammatory and antioxidant benefits.",
    "Cloves plant": "The aromatic flower buds of a tree native to the Maluku Islands in Indonesia. \
Cloves are used as a spice for flavor and aroma, particularly in baking and curries. \
They are rich in eugenol, an antioxidant and anesthetic compound. \
Historically, cloves were a valuable commodity in the spice trade.",
    "Cocoa beans plant": "The seeds of the cacao tree, native to the tropical regions of the Americas. \
Cocoa is rich in flavonoids, which support heart health. \
The beans are fermented, dried, and roasted to produce cocoa powder and chocolate. \
West Africa is now the largest producer of cocoa.",
    "Coconuts plant": "The fruit of the coconut palm, grown in tropical and subtropical coastal regions. \
Coconuts provide healthy fats, fiber, and electrolytes. \
Used for their water, oil, milk, and dried flesh in both food and cosmetics. \
Almost every part of the tree is useful, earning it the nickname 'the tree of life'.",
    "Coffee (green) plant": "A flowering shrub native to tropical Africa, particularly Ethiopia. \
Coffee beans, the plant's seeds, are a major source of caffeine. \
Green coffee beans are unroasted and used for health supplements for their chlorogenic acid content. \
After roasting, they are ground and brewed to create the popular beverage.",
    "Collards plant": "A leafy green vegetable in the cabbage family, cultivated globally. \
Rich in vitamins K, A, and C, and a good source of dietary fiber. \
Traditionally boiled or steamed and served as a side dish, especially in Southern US cuisine. \
The leaves do not form a head like cabbage, instead growing as a loose bunch.",
    "Cotton lint plant": "A shrub native to tropical and subtropical regions worldwide. \
The cotton fiber, or lint, is a soft, fluffy staple fiber that grows in a boll. \
It is used primarily to produce textiles and is the most widely used natural fiber. \
The seeds are also used for cotton seed oil and animal feed.",
    "Cranberries plant": "A low-growing evergreen shrub native to North America. \
Cranberries are rich in antioxidants and proanthocyanidins, which may help prevent UTIs. \
Consumed in sauces, juices, and dried snacks due to their tart flavor. \
The berries grow on vines and a field is flooded for the harvest.",
    "Cucumbers and gherkins plant": "Vining plants native to India, now cultivated globally. \
Cucumbers are high in water content, helping with hydration. \
Eaten fresh in salads, or pickled to make gherkins. \
They are botanically a fruit but used as a vegetable.",
    "Dates plant": "The fruit of the date palm tree, native to the Middle East and North Africa. \
Dates are rich in natural sugars, fiber, and minerals like potassium. \
Eaten fresh or dried as a sweet snack, or used in baking. \
They are a traditional staple food and a significant part of desert cultures.",
    "Dry beans plant": "A group of legumes that are grown for their edible seeds. \
Beans are a great source of plant-based protein, fiber, and iron. \
A staple food worldwide, used in countless cuisines from stews to salads. \
Varieties include kidney, pinto, black, and navy beans.",
    "Dry peas plant": "A legume cultivated since ancient times, native to the Middle East. \
Dry peas are an excellent source of protein, fiber, and essential vitamins. \
Used in soups, stews, and ground into flour for baking. \
They are often grown as a cover crop to improve soil health.",
    "Durian plant": "A tropical fruit tree native to Southeast Asia, known as the 'king of fruits'. \
The fruit is rich in fiber, B vitamins, and antioxidants. \
It is famous for its large size, thorny exterior, and strong odor. \
Despite the smell, its creamy flesh is highly prized in its native region.",
    "Eggplants (Aubergines) plant": "A species in the nightshade family native to India and Sri Lanka. \
Eggplants are high in fiber and a variety of nutrients, including antioxidants. \
Used extensively in Mediterranean, Middle Eastern, and Asian cuisines. \
They can be grilled, roasted, fried, or stewed.",
    "Endive plant": "A leafy vegetable belonging to the chicory family, native to the Mediterranean. \
It is a good source of fiber, vitamin K, and folate. \
The leaves have a slightly bitter taste and are used in salads or braised dishes. \
Blanched endive has a less bitter and more delicate flavor.",
    "Fava bean plant": "A species of vetch in the pea family, widely cultivated in the Middle East and North Africa. \
Fava beans are high in protein, fiber, and a good source of folate. \
Used in stews, salads, and as a popular street food in many countries. \
They are one of the most ancient cultivated crops.",
    "Figs plant": "The fruit of the fig tree, native to the Mediterranean and West Asia. \
Figs are a good source of fiber, potassium, and calcium. \
Eaten fresh or dried as a sweet snack, or used in jams and desserts. \
They are a symbol of abundance and prosperity in many cultures.",
    "Flax fiber and tow plant": "A plant cultivated for its seeds and textile fibers, native to the region from the eastern Mediterranean to India. \
The fibers are used to produce linen textiles, known for their strength and absorbency. \
Flax fiber is a key component in high-quality paper and industrial materials. \
It's one of the oldest cultivated plants for fiber.",
    "Flaxseed (Linseed) plant": "A crop grown for its seeds, native to Europe and Asia. \
Flaxseeds are a powerhouse of Omega-3 fatty acids, fiber, and lignans. \
They are consumed whole, ground, or as an oil for health benefits. \
Used in baking, smoothies, and as a dietary supplement for digestive health.",
    "Fonio plant": "A grain native to West Africa, known for its small size and quick growth. \
Fonio is a gluten-free grain, rich in amino acids and fiber. \
Used to make porridge, couscous, and is a staple food in many African diets. \
It's considered a 'future food' due to its resilience to poor soil and climate change.",
    "Garlic plant": "A perennial herb in the onion family, native to Central Asia. \
Garlic is a natural antibiotic and anti-inflammatory, used for its health benefits. \
A staple in cuisines worldwide, used to add a pungent, savory flavor. \
It has been used for centuries for medicinal and culinary purposes.",
    "Ginger plant": "A flowering plant whose rhizome is widely used as a spice and folk medicine. \
Ginger has anti-inflammatory properties and is known to aid digestion. \
Used fresh, dried, powdered, or as an oil in a variety of dishes. \
It is a key ingredient in Asian and Indian cuisines.",
    "Gooseberries plant": "A flowering shrub native to Europe, Asia, and North America. \
Gooseberries are rich in vitamin C, antioxidants, and dietary fiber. \
The berries are used in jams, pies, and sauces. \
They can be green, yellow, red, or purple and vary from sweet to tart.",
    "Grapes plant": "A flowering plant, the fruit of which grows in clusters on a vine. \
Grapes are a good source of vitamin K, C, and antioxidants. \
Eaten fresh, dried as raisins, or fermented to make wine. \
They are one of the most widely cultivated fruits in the world.",
    "Groundnuts (Peanuts) plant": "A legume crop grown for its edible seeds, originating in South America. \
Peanuts are high in protein, healthy fats, and niacin (B3). \
Used in snacks, peanut butter, oil, and a wide variety of dishes. \
Unlike tree nuts, they grow underground.",
    "Guarana plant": "A climbing plant native to the Amazon basin. \
The seeds contain high concentrations of caffeine, much more than coffee beans. \
Used as a stimulant in energy drinks and supplements for a natural energy boost. \
It's a staple in Brazilian culture and traditional medicine.",
    "Guavas plant": "A tropical fruit tree native to Mexico, Central America, and South America. \
Guavas are exceptionally rich in vitamin C and dietary fiber. \
Eaten fresh, or used in juices, jams, and desserts. \
The pink variety has a sweet, musky flavor while the white is more tart.",
    "Habanero pepper plant": "A hot chili pepper, a variety of the Capsicum chinense species. \
Habaneros are known for their extreme heat and fruity, citrus-like flavor. \
They are a good source of vitamin C and capsaicin. \
Native to the Amazon, it is now widely cultivated in Mexico's Yucatán Peninsula.",
    "Hazelnuts plant": "The nut of the hazel tree, native to temperate regions of the Northern Hemisphere. \
Hazelnuts are rich in healthy fats, protein, and vitamin E. \
Used in confectionery like Nutella, baked goods, and as a snack. \
Turkey is the world's largest producer of hazelnuts.",
    "Hemp plant": "A variety of the Cannabis sativa plant species cultivated for its fiber and seeds. \
Hemp seeds are a source of protein, Omega-3s, and fiber. \
The strong fibers are used to make textiles, rope, paper, and building materials. \
It's a highly sustainable and versatile crop with numerous applications.",
    "Hen eggs (shell weight) plant": "This is not a plant, but a food product from hens. \
Eggs are a complete protein source, containing all nine essential amino acids. \
They are used in baking, cooking, and as a standalone meal. \
The shell itself, made of calcium carbonate, is not consumed but can be used as a soil amendment.",
    "Horseradish plant": "A perennial plant of the mustard family, native to Southeast Europe and West Asia. \
The root is grated to produce a pungent condiment, known for its sinus-clearing heat. \
It contains compounds that may have antibacterial and anti-inflammatory properties. \
Traditionally used in sauces for roast beef and other meats.",
    "Jackfruit plant": "A tropical tree native to the Western Ghats of India. \
The fruit is the largest tree-borne fruit in the world. \
Young jackfruit is a popular meat substitute due to its fibrous, stringy texture when cooked. \
It is rich in vitamins, minerals, and dietary fiber.",
    "Jute plant": "A vegetable fiber plant primarily cultivated in Bangladesh and India. \
Jute fibers are strong, shiny, and used to make burlap, ropes, and other coarse textiles. \
It's one of the most affordable natural fibers, second only to cotton in production. \
Jute helps improve soil fertility and is a key cash crop in its growing regions.",
    "Kale plant": "A leafy green vegetable belonging to the cabbage family. \
Kale is a nutritional powerhouse, packed with vitamins K, A, and C, and calcium. \
It is used in salads, smoothies, and as a side dish, often roasted or sautéed. \
Its sturdy leaves hold up well to cooking and massage.",
    "Kohlrabi plant": "A biennial vegetable in the cabbage family, cultivated for its swollen, turnip-like stem. \
It is high in vitamin C, fiber, and potassium. \
Eaten raw in salads, roasted, or steamed. \
Its name means 'cabbage turnip' in German.",
    "Leeks plant": "A vegetable in the onion family, native to Central Asia. \
Leeks are rich in vitamins K and A, and have a mild, sweet flavor. \
Used in soups, stews, and a variety of European dishes. \
They are a staple in Welsh cuisine and a national symbol of Wales.",
    "Lemons and limes plant": "Citrus trees cultivated in tropical and subtropical regions worldwide. \
These fruits are an excellent source of vitamin C and antioxidants. \
Used for their sour juice and aromatic zest in cooking, beverages, and desserts. \
Lemons and limes were historically used to prevent scurvy on long sea voyages.",
    "Lentils plant": "A small legume cultivated for its lens-shaped seeds, native to the Near East. \
Lentils are a great source of protein, fiber, and iron. \
A key ingredient in stews, soups, and curries worldwide. \
They are one of the earliest domesticated crops.",
    "Lettuce and chicory plant": "Leafy vegetables grown globally. \
Lettuce is known for its high water content and vitamins A and K. \
Chicory, often more bitter, is a good source of fiber and folate. \
Both are staples in salads and sandwiches.",
    "Lima bean plant": "A legume native to Central and South America. \
Lima beans are a good source of plant-based protein, fiber, and folate. \
They are eaten fresh or dried and used in soups, stews, and side dishes. \
The name comes from Lima, Peru, where they were widely cultivated.",
    "Longan plant": "A tropical fruit tree native to Southeast Asia. \
The fruit is small, round, and has a translucent, sweet, juicy flesh. \
Longan is rich in vitamin C and potassium. \
Its name means 'dragon eye' in Chinese due to the white flesh and black seed.",
    "Lupins plant": "A legume grown for its seeds, found in the Mediterranean and Americas. \
Lupin beans are very high in protein and fiber, and low in fat. \
Used in snacks, flour, and as a protein supplement in food products. \
They are also grown as a green manure crop to improve soil fertility.",
    "Lychee plant": "A tropical fruit tree native to Southeast Asia. \
The fruit has a rough, pink-red skin and sweet, juicy, translucent flesh. \
Lychees are an excellent source of vitamin C. \
They are a popular summer fruit, eaten fresh or used in desserts and beverages.",
    "Maize (Corn) plant": "A cereal grain domesticated by indigenous peoples in southern Mexico. \
Corn is a major source of carbohydrates, fiber, and vitamins. \
Used for human consumption (on the cob, cornmeal, popcorn), animal feed, and ethanol production. \
It is one of the most widely grown crops globally.",
    "Mandarins, clementines, satsumas plant": "A group of citrus trees native to China. \
These fruits are rich in vitamin C and antioxidants. \
Eaten fresh due to their easy-to-peel skin and sweet, less acidic flavor. \
They are a popular fruit for snacking and juicing.",
    "Mangoes, mangosteens, guavas plant": "Tropical fruit trees native to Southeast Asia and the Americas. \
Mangoes are high in vitamins A and C. Mangosteens are rich in antioxidants. Guavas are packed with vitamin C. \
These fruits are eaten fresh, used in juices, jams, and desserts. \
They are staples in tropical diets and a vital export crop.",
    "Maracuja(Passionfruit) plant": "A vine species of passion flower, native to Brazil. \
The fruit is rich in vitamin A, vitamin C, and dietary fiber. \
The tart, aromatic pulp is used in juices, desserts, and cocktails. \
Its name comes from the Christian symbolism of its flower's structure.",
    "Millet plant": "A group of small-seeded grasses cultivated worldwide as cereal crops. \
Millet is gluten-free and a good source of protein, fiber, and magnesium. \
A staple food in Africa and Asia, used to make porridge, flatbreads, and beer. \
It is known for its ability to grow in arid conditions.",
    "Mint plant": "An aromatic perennial herb, native to Europe and Asia. \
Mint is known for its refreshing flavor and can aid in digestion. \
Used in teas, beverages, desserts, and savory dishes. \
It is widely used in traditional medicine for its soothing properties.",
    "Mung bean plant": "A legume species native to the Indian subcontinent. \
Mung beans are a great source of plant-based protein, fiber, and B vitamins. \
Used in a variety of dishes, from curries and soups to sprouted salads. \
They are a key part of Indian and Chinese cuisines.",
    "Mustard greens plant": "A leafy vegetable of the mustard family. \
They are rich in vitamins K, C, and A, and a good source of calcium. \
Used in stir-fries, soups, and curries, especially in Asian and African cuisines. \
The leaves have a pungent, peppery flavor that mellows with cooking.",
    "Mustard seeds plant": "The seeds of the mustard plant, native to temperate regions of Europe. \
Mustard seeds are rich in minerals like selenium and magnesium. \
They are used whole or ground to make the condiment mustard, and in cooking for their pungent flavor. \
Mustard is a key ingredient in many pickling spice blends and curries.",
    "Navy bean plant": "A variety of the common bean, originally cultivated in the Americas. \
Navy beans are high in fiber, folate, and protein. \
They are a staple in baked bean dishes, soups, and stews. \
The name 'navy bean' comes from their use as a food by the U.S. Navy.",
    "Oats plant": "A cereal grain cultivated for its seeds, native to Europe and Asia. \
Oats are a great source of soluble fiber, which can help lower cholesterol. \
Used to make oatmeal, granola, and in baking. \
They are a common breakfast food and are also used as animal feed.",
    "Oil palm fruit plant": "A palm tree native to West and Southwest Africa. \
The fruit is used to produce palm oil, a major global commodity. \
Palm oil is a source of vitamin E and is used in a vast range of food and consumer products. \
It is an extremely high-yielding oil crop.",
    "Okra plant": "A flowering plant in the mallow family, native to Africa. \
Okra is rich in fiber, vitamin C, and folate. \
The edible green seed pods are used in stews, soups, and fried dishes. \
Its mucilage, or 'slime', is used to thicken gumbo and other sauces.",
    "Olives plant": "The fruit of the olive tree, native to the Mediterranean basin. \
Olives are a source of healthy monounsaturated fats and antioxidants. \
Eaten whole after curing, or pressed to produce olive oil. \
Olive oil is a cornerstone of the healthy Mediterranean diet.",
    "Onions (dry) plant": "A bulb vegetable in the allium family, native to Central Asia. \
Onions are a good source of vitamin C and prebiotics, which support gut health. \
Used as a flavor base in almost every cuisine worldwide. \
They are known for their pungent taste and aroma when chopped.",
    "Oranges plant": "A citrus fruit tree, a hybrid of mandarin and pomelo, native to Southeast Asia. \
Oranges are a renowned source of vitamin C, boosting the immune system. \
Eaten fresh, juiced, or used in desserts and marinades. \
Brazil and the United States are major orange producers.",
    "Oregano plant": "A perennial herb in the mint family, native to the Mediterranean region. \
Oregano is rich in antioxidants and has antibacterial properties. \
Used dried or fresh to season pizzas, pastas, and meats in Italian and Mediterranean cuisines. \
Its name means 'mountain joy' in Greek.",
    "Papayas plant": "A tropical fruit tree native to Mexico and Central America. \
Papayas are rich in vitamins C and A, and the enzyme papain, which aids digestion. \
The sweet, orange flesh is eaten fresh or used in smoothies. \
Green, unripe papayas are used as a vegetable in savory dishes.",
    "Parsley plant": "A biennial herb native to the central Mediterranean region. \
Parsley is packed with vitamins K, C, and A. \
Used as a garnish or flavor enhancer in a wide variety of dishes worldwide. \
It is known for its fresh, clean taste and is a staple in Middle Eastern cuisine.",
    "Peaches and nectarines plant": "Fruit trees native to China. \
Both are rich in vitamins A and C, and antioxidants. \
Peaches have a fuzzy skin, while nectarines have smooth skin. \
They are enjoyed fresh, grilled, or used in pies and jams.",
    "Peas (Green) plant": "A small spherical legume, a cool-season crop native to the Middle East. \
Green peas are a good source of protein, fiber, and vitamins. \
They are eaten fresh, frozen, or canned and used in soups, salads, and side dishes. \
Peas are a classic example of a legume grown for its young seeds.",
    "Persimmons plant": "The edible fruit of a tree, native to East Asia. \
Persimmons are rich in vitamins A and C, and dietary fiber. \
The fruit can be eaten fresh when ripe, or dried. \
They are known for their sweet, honey-like flavor when fully mature.",
    "Pine nuts plant": "The edible seeds of pine trees, native to Europe, Asia, and North America. \
Pine nuts are a good source of healthy fats, protein, and magnesium. \
Used in pesto, salads, and as a garnish in Mediterranean and Middle Eastern cuisines. \
Harvesting them can be a laborious process.",
    "Pineapples plant": "A tropical fruit native to South America. \
Pineapple is rich in vitamin C and the digestive enzyme bromelain. \
Eaten fresh, canned, juiced, or used in savory dishes. \
A single pineapple is actually a fusion of many individual flowers.",
    "Pinto bean plant": "A variety of the common bean, native to the Americas. \
Pinto beans are high in fiber, protein, and folate. \
A staple in Mexican and Tex-Mex cuisine, often used in refried beans. \
The name 'pinto' refers to its mottled, spotted appearance.",
    "Pistachios plant": "The edible seed of a tree native to Central Asia and the Middle East. \
Pistachios are a great source of protein, fiber, and healthy fats. \
Used in snacks, ice cream, and a variety of desserts. \
Iran and the United States are the top producers.",
    "Plantains plant": "A member of the banana family, native to Southeast Asia. \
Plantains are starchier and lower in sugar than bananas. \
A cooking staple in tropical regions, they are typically fried, roasted, or boiled. \
They are a major source of carbohydrates for millions of people.",
    "Pomegranates plant": "A fruit-bearing shrub native to the region from Iran to Northern India. \
Pomegranate seeds (arils) are rich in fiber, vitamin K, and antioxidants. \
Used in juices, salads, and desserts, or eaten fresh. \
The fruit's juice is known for its vibrant red color and tart flavor.",
    "Potatoes plant": "A starchy tuber, a root vegetable native to the Andes of South America. \
Potatoes are a good source of potassium and vitamin C. \
A major staple food crop worldwide, prepared in countless ways. \
They were first domesticated by indigenous peoples in modern-day Peru.",
    "Pumpkins, squash and gourds plant": "A family of plants native to North America. \
These vegetables are rich in vitamins A and C, and dietary fiber. \
Used in soups, pies, roasted dishes, and stews. \
They are known for their hard rinds and long storage life.",
    "Quinoa plant": "A flowering plant native to the Andes mountains of South America. \
Quinoa is a gluten-free pseudocereal, high in protein and all nine essential amino acids. \
Used as a grain replacement in salads, porridges, and side dishes. \
It's considered a 'superfood' for its nutritional profile.",
    "Radishes and similar roots plant": "Root vegetables in the mustard family, native to Asia. \
Radishes are a good source of vitamin C and have a peppery taste. \
Eaten raw in salads or pickled. \
They are a fast-growing crop, often ready for harvest in a few weeks.",
    "Rambutan plant": "A tropical fruit tree native to Southeast Asia. \
The fruit has a hairy, red or yellow rind and sweet, translucent flesh. \
Rambutan is rich in vitamin C and iron. \
Its name means 'hair' in Malay, a reference to the fruit's appearance.",
    "Rapeseed (Canola) plant": "A bright-yellow flowering plant cultivated mainly in Canada, China, and India. \
The seeds are used to produce canola oil, a heart-healthy cooking oil. \
Canola oil is low in saturated fat and high in monounsaturated fats. \
The plant's meal is also used as animal feed.",
    "Raspberries plant": "The fruit of a flowering plant in the rose family, native to Europe and North Asia. \
Raspberries are rich in antioxidants, vitamin C, and fiber. \
They are enjoyed fresh, in jams, juices, and desserts. \
The fruit is an aggregate of many small 'drupelets'.",
    "Rice (Paddy) plant": "A cereal grain, a staple food for a large part of the world's population. \
Rice is a major source of carbohydrates and a key energy provider. \
It is cultivated in flooded fields known as paddies, mainly in Asia. \
There are thousands of rice varieties, each with unique characteristics.",
    "Rosemary plant": "A fragrant, evergreen herb native to the Mediterranean region. \
Rosemary is rich in antioxidants and has anti-inflammatory properties. \
Used fresh or dried to flavor roasted meats, vegetables, and breads. \
It is a symbol of remembrance in many cultures.",
    "Rubber (natural) plant": "A tree native to the Amazon rainforest, grown for its milky sap. \
The sap, or latex, is harvested and processed to make natural rubber. \
Natural rubber is used in tires, gloves, and other elastic products. \
The plant's cultivation is a major industry in Southeast Asia.",
    "Rye plant": "A cereal grain, a member of the wheat family, native to East-Central Turkey. \
Rye is high in fiber and minerals like manganese. \
Used to make flour for dark breads, whiskey, and animal feed. \
It is very tolerant of cold climates and poor soil conditions.",
    "Saffron plant": "A flower whose threads are the world's most expensive spice by weight. \
Saffron is known for its distinct flavor, aroma, and vibrant color. \
It is rich in antioxidants and is used in a variety of dishes, especially paella and biryani. \
Each flower produces only three stigmas, which are hand-picked.",
    "Sage plant": "A perennial herb in the mint family, native to the Mediterranean. \
Sage is rich in antioxidants and is traditionally used to improve cognitive function. \
Used to flavor roasted meats, stuffings, and sauces. \
Its name comes from the Latin 'salvare', meaning 'to save', referencing its medicinal uses.",
    "Scallions plant": "A vegetable in the onion family, also known as green onions or spring onions. \
They are a good source of vitamins K and C. \
Used for their mild, fresh onion flavor in salads, soups, and stir-fries. \
Both the green stalk and the white bulb are edible.",
    "Sorghum plant": "A cereal grain native to Africa, now grown globally in warmer climates. \
Sorghum is gluten-free and a good source of fiber and protein. \
Used for human consumption as a grain or syrup, and for animal feed. \
It is a highly drought-resistant crop.",
    "Soursop plant": "A fruit tree native to the tropical regions of the Americas. \
The fruit has a green, prickly skin and a creamy, white, fragrant flesh. \
Soursop is rich in vitamin C and fiber. \
It is used in juices, ice creams, and desserts.",
    "Soybeans plant": "A species of legume native to East Asia. \
Soybeans are a complete protein source, containing all nine essential amino acids. \
They are processed into tofu, soy milk, soy sauce, and used as a source of oil. \
They are one of the most important global agricultural crops.",
    "Spinach plant": "A leafy green flowering plant native to Central and Western Asia. \
Spinach is packed with iron, vitamin K, vitamin A, and folate. \
Eaten raw in salads or cooked in a variety of dishes. \
It is known for wilting significantly when cooked.",
    "Starfruit plant": "A fruit tree native to Southeast Asia. \
The fruit, when cut in cross-section, resembles a five-pointed star. \
Starfruit is high in vitamin C and antioxidants. \
Its taste is a mix of apple, grape, and citrus.",
    "Strawberries plant": "A hybrid species of fruit, cultivated globally. \
Strawberries are an excellent source of vitamin C and antioxidants. \
Eaten fresh, used in jams, desserts, and smoothies. \
The plant is a member of the rose family.",
    "Sugar beet plant": "A plant grown for its root, which contains a high concentration of sucrose. \
The sugar extracted from sugar beets is chemically identical to that from sugar cane. \
It is a major source of sugar production in temperate climates. \
The remaining pulp is used for animal feed.",
    "Sugar cane plant": "A tall grass native to tropical South and Southeast Asia. \
The stalks contain a high concentration of sucrose, which is extracted to produce sugar. \
Sugar cane is the world's largest crop by production volume. \
It is also used to produce molasses and rum.",
    "Sunflower seeds plant": "The seeds of the sunflower, a large flowering plant native to North America. \
Sunflower seeds are a great source of vitamin E, healthy fats, and protein. \
Used in snacks, salads, breads, and pressed to produce sunflower oil. \
The flowers track the sun's movement throughout the day.",
    "Sweet potatoes plant": "A starchy, sweet-tasting root vegetable native to the tropical Americas. \
Sweet potatoes are an excellent source of beta-carotene, which the body converts to vitamin A. \
Used in both savory and sweet dishes, from fries to pies. \
They are a major source of carbohydrates in many parts of the world.",
    "Swiss chard plant": "A leafy green vegetable in the beet family, native to the Mediterranean region. \
Swiss chard is a powerhouse of vitamins K, A, and C, and minerals like magnesium. \
The leaves and stalks are used in salads, sautéed dishes, and soups. \
It's a very hardy and high-yielding crop.",
    "Tamarind plant": "A legume tree native to tropical Africa. \
The pulp of the fruit is a source of antioxidants and can aid in digestion. \
It is used in sauces, chutneys, and beverages for its distinctive sweet and sour flavor. \
A key ingredient in many Asian, Latin American, and Caribbean cuisines.",
    "Taro (cocoyam) plant": "A tropical plant grown primarily for its starchy corms, native to Southeast Asia and Southern India. \
Taro is a good source of fiber, potassium, and vitamins. \
The corms are boiled, fried, or mashed and are a staple food in many tropical cultures. \
The leaves are also edible when cooked.",
    "Tea plant": "An evergreen shrub or small tree native to East Asia. \
The leaves are processed to create various types of tea, a major source of antioxidants. \
Used as a beverage globally, appreciated for its flavor and stimulating effects. \
The major producers are China, India, and Sri Lanka.",
    "Teff plant": "A cereal grain native to the Horn of Africa, especially Ethiopia. \
Teff is a gluten-free grain, rich in protein, iron, and calcium. \
Used to make injera, a traditional Ethiopian flatbread, and porridges. \
It's a tiny grain but a nutritional giant.",
    "Thyme plant": "A fragrant, evergreen herb in the mint family, native to the Mediterranean. \
Thyme is rich in antioxidants and has antiseptic properties. \
Used to flavor meats, soups, stews, and vegetables in many cuisines. \
It's a staple in both French and Middle Eastern cooking.",
    "Tomatoes plant": "A flowering plant in the nightshade family, native to western South America. \
Tomatoes are a good source of vitamin C, potassium, and lycopene, an antioxidant. \
Used as a fruit or vegetable in sauces, salads, and countless dishes. \
They are botanically a fruit but culinarily a vegetable.",
    "Triticale plant": "A hybrid grain of wheat and rye. \
Triticale is high in protein and fiber, combining the yield of wheat with the hardiness of rye. \
Used in animal feed, baked goods, and some breakfast cereals. \
It was first bred in Scotland and Sweden in the 19th century.",
    "Turmeric plant": "A flowering plant in the ginger family, native to Southeast Asia and India. \
The root is used as a spice and traditional medicine due to its anti-inflammatory properties. \
It adds a distinctive yellow color and earthy flavor to curries. \
Curcumin is the active compound in turmeric.",
    "Turnip greens plant": "The leafy tops of the turnip plant. \
They are an excellent source of vitamins K and A, and calcium. \
Often cooked and served as a side dish in Southern US cuisine. \
The greens have a slightly bitter flavor that becomes milder with cooking.",
    "Vanilla beans plant": "A species of orchid, native to Mexico. \
The fermented and cured pods, or beans, are used as a flavoring. \
Vanilla is rich in antioxidants and has a comforting aroma. \
It is one of the most labor-intensive crops to produce.",
    "Walnuts plant": "The nut of a tree native to the temperate regions of the Northern Hemisphere. \
Walnuts are a great source of Omega-3 fatty acids, antioxidants, and protein. \
Used in snacks, baking, salads, and as a topping. \
They are known for their distinct brain-like shape.",
    "Watermelons plant": "A vine-like flowering plant native to West Africa. \
Watermelon is over 90% water, making it excellent for hydration. \
The sweet, juicy fruit is a popular summer snack. \
It is a good source of vitamin C and A.",
    "Wheat plant": "A cereal grain cultivated globally, originating in the Middle East. \
Wheat is a major source of carbohydrates, protein, and dietary fiber. \
The grain is ground into flour for bread, pasta, and baked goods. \
It is the world's most widely grown cereal crop.",
    "Yams plant": "A starchy tuberous root vegetable native to Africa and Asia. \
Yams are a major source of carbohydrates, fiber, and potassium. \
A staple food in many tropical regions, used in soups, stews, and roasted dishes. \
They should not be confused with sweet potatoes.",
}
# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Crop Classifier", layout="centered")
st.title("🌱🌿 Crop_Classifier 🌿🌱")
st.markdown(
    "Upload a crop image and the model will predict which crop it is. A short description of the predicted crop will be shown below."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and run model
    input_tensor = transform(image).unsqueeze(0)

    # Ensure model and tensor on CPU/GPU consistent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_name = idx_to_class[predicted.item()]

    st.success(f"Predicted Crop: **{class_name}**")

    # Display description
    description = plant_descriptions.get(class_name, None)
    if description:
        st.info(description)
    else:
        st.warning(
            "Description not available for this crop. You can add it to the plant_descriptions dictionary in app.py."
        )
