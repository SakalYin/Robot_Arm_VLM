import pandas as pd
import random
from PIL import Image
import matplotlib.pyplot as plt

def read_data(path, cols):
    data = []

    with open(path, 'r') as file:
        for line in file.readlines():
            data.append(line.split('\t'))
    data = pd.DataFrame(data, columns=cols)
    return data

def plot_images_side_by_side(image_paths):
    if len(image_paths) != 3:
        raise ValueError("Exactly 3 image paths are required.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    for ax, img_path in zip(axes, image_paths):
        img = Image.open(img_path)  # Read the image
        ax.imshow(img)  # Display the image
        ax.axis('off')  # Hide axis
        ax.set_title(img_path.split("/")[-1])  # Set filename as title

    plt.tight_layout()
    plt.show()

def remove_static_eff(data):
    index = []

    for i in range(2, len(data)):
        if data.object[i] == data.object[i-2]:
            if data.cur_pose[i] == data.cur_pose[i-2]:
                index.append(i)
                index.append(i-1)
                index.append(i-2)

    return index

def prepare_data(data_folder, camera_loc, drop_static=True, reset_frac=0.1, float_round=5):

    # load and combine data + eff data
    data = read_data(data_folder+'/label.txt', ['path', 'object', 'pose', 'orient'])
    eff = read_data(data_folder+'/eff_label.txt', ['path', 'cur_pose', 'cur_orient'])
    data = pd.merge(data, eff, on='path')

    # get images name
    data['images'] = data['path'].apply(lambda x: x.split('/')[-1])

    # drop static eff data
    if drop_static:
        static_eff_idx = remove_static_eff(data)
        print(f'Removed {len(static_eff_idx)} static images')
        data = data.drop(static_eff_idx).reset_index(drop=True).drop(columns=['path', 'cur_pose', 'cur_orient'])

    # sample some data for reset action
    data_sampled = data.sample(frac=reset_frac, random_state=42)
    data_sampled['object'] = 'reset'
    data_sampled['pose'] = "0.32,-0.65897,1.5002"
    data_sampled['orient'] = "0.70711,0.00060,0.00068,0.70711"
    data = pd.concat([data, data_sampled], ignore_index=True)

    # randomize the prompts
    data['prompt'] = None
    for i in range(len(data)):
        if 'reset' in data.loc[i, 'object']:
            data.loc[i, 'prompt'] = random.choice(reset_prompts)
        if 'coke_can' in data.loc[i, 'object']:
            data.loc[i, 'prompt'] = random.choice(coke_prompts)
        if 'samsung_j8_black' in data.loc[i, 'object']:
            data.loc[i, 'prompt'] = random.choice(smartphone_prompts)
        if 'peach' in data.loc[i, 'object']:
            data.loc[i, 'prompt'] = random.choice(peach_prompts)
        if 'bowl' in data.loc[i, 'object']:
            data.loc[i, 'prompt'] = random.choice(bowl_prompts)
        if 'plastic_cup' in data.loc[i, 'object']:
            data.loc[i, 'prompt'] = random.choice(plastic_cup_prompts)
        if 'strawberry' in data.loc[i, 'object']:
            data.loc[i, 'prompt'] = random.choice(strawberry_prompts)
    # data['prompt'] = data['prompt'].apply(lambda x: x + f'<camera>{camera_loc}</camera>')

    # Round the floats
    pos = data.pose[i].split(',')
    pos = ','.join([str(round(float(num), float_round)) for num in pos])

    # Prepare the model outputs
    data['output'] = None
    for i in range(len(data)):
        obj = data.object[i]
        pos = data.pose[i].split(',')
        pos = ','.join([str(round(float(num), 5)) for num in pos])
        
        ori = data.orient[i].split(',')
        ori = ','.join([str(round(float(num), 5)) for num in ori])
        data.loc[i, 'output'] = f"<obj>{obj}</obj> <pose>{pos}</pose> <orient>{ori}</orient>"
    
    # Redefine the paths and shuffle the dataset
    data['images'] = data['images'].apply(lambda x: data_folder+'/' + x)

    # export
    data[['images', 'prompt', 'output']].sample(frac=1).to_json(data_folder+'.json', orient='records', indent=4, force_ascii=False)

bowl_prompts = [
    "I need a holder for my ramen.",
    "I want to put some soup in something.",
    "I need a container for my cereal.",
    "I want to mix some ingredients together.",
    "Where can I place my hot noodles?",
    "I need something to hold my fruit.",
    "I’m looking for something to serve rice in.",
    "I want to eat some salad, but I need a container.",
    "I need something deep to hold liquid food.",
    "I want to scoop some food into a dish.",
    "There's some hot soup, but I have nothing to put it in.",
    "This meal would be easier to eat with something to hold it.",
    "I should get something deep enough to hold my food.",
    "Something to keep my noodles from spilling would be great.",
    "I need to serve this dish properly.",
    "Where can I place this fruit so it doesn’t roll away?",
    "I’m about to mix some ingredients—what can I use?",
    "I need to separate this portion from the rest.",
    "This dish is too messy to eat without a proper container.",
    "I have some snacks, but I need a place to put them.",
    "This soup is too hot to hold—I need something to put it in.",
    "I need something that can hold both solid and liquid food.",
    "A deep dish would be perfect for this meal.",
    "I want to serve my salad properly instead of just piling it up.",
    "I should grab something spacious enough to hold a good portion of rice.",
    "Pouring cereal without something to hold it would be a disaster.",
    "I need something that keeps my food from spilling over the edges.",
    "Mixing these ingredients without a proper container will make a mess.",
    "A wide and deep container is exactly what I need right now.",
    "I want to keep all my food together instead of it spreading out.",
    "I need a bowl to hold my soup before it gets cold.",
    "A bowl would be perfect for mixing this salad.",
    "Where’s a bowl I can use to serve this dish?",
    "This ramen needs a bowl so I can eat it properly.",
    "A cold Coke would be refreshing right now.",
    "I should grab a Coke to go with my meal.",
    "A can of Coke would hit the spot.",
    "Where’s the Coke? I could use something fizzy."
]

coke_prompts = [
    "I'm thirsty for some sweet drinks.",
    "I need a cold refreshment.",
    "I want something fizzy to drink.",
    "I'm craving a carbonated beverage.",
    "I need something sugary to quench my thirst.",
    "I’m looking for a soft drink.",
    "I want to grab a canned soda.",
    "I need something with caffeine but cold.",
    "Where can I find a refreshing drink?",
    "I need a cold beverage to cool down.",
    "A refreshing drink right now would be perfect.",
    "Something fizzy would go well with this meal.",
    "I could use something cold and sweet to drink.",
    "A carbonated drink sounds like a good idea.",
    "It's been a long day—I should grab something refreshing.",
    "A soft drink would really hit the spot.",
    "A cold beverage would make this meal complete.",
    "Is there something sugary to drink around here?",
    "This meal needs a nice chilled drink to go with it.",
    "Something in a bottle with bubbles would be nice.",
    "Something cold and fizzy would go well with my meal.",
    "I feel like drinking something that has a little kick to it.",
    "A canned drink would be perfect right now.",
    "I want a drink that gives me a quick burst of energy.",
    "A sweet, carbonated drink would help me stay refreshed.",
    "This weather is so hot—I could use a cold, bubbly refreshment.",
    "I need something that satisfies my craving for something sugary.",
    "A drink with some fizz would really hit the spot.",
    "A dark, sweet beverage would be the perfect pairing for this meal.",
    "I should grab a bottle of something with a recognizable taste."
]

peach_prompts = [
    "I want something sweet and juicy to eat.",
    "I'm craving a soft fruit.",
    "I need a healthy snack.",
    "I’m looking for something fresh and round to eat.",
    "I need a piece of fruit that’s not too big.",
    "Where is the fresh fruit?",
    "I need something small and nutritious.",
    "I want something refreshing to eat.",
    "I could go for something naturally sweet.",
    "A soft, fresh fruit would be perfect.",
    "Where’s something I can grab for a light snack?",
    "Something refreshing and slightly fuzzy to the touch would be nice.",
    "This yogurt could use some fresh fruit in it.",
    "I should grab something sweet but not too heavy.",
    "A piece of fruit sounds like a good idea.",
    "I feel like eating something fresh and juicy.",
    "A small, round fruit would be just right.",
    "I want to eat something that has a soft texture and sweet taste.",
    "A piece of fruit with a firm outside and juicy inside sounds great.",
    "I should find something I can bite into and enjoy.","This would be even better with something naturally sweet and juicy.",
    "I could go for something with a fragrant aroma and soft flesh.",
    "A fruit with a pit in the middle would be a nice snack.",
    "Something round and slightly fuzzy sounds like the perfect treat.",
    "I should grab something that tastes best when perfectly ripe.",
    "A peach would be a great snack right now.",
    "I should grab a peach for something naturally sweet.",
    "This meal would taste better with a peach on the side.",
    "Where’s a ripe peach? I’m craving something juicy."
]

plastic_cup_prompts = [
    "I need something to pour my water into.",
    "Where can I pour my juice?",
    "I need a container for my drink.",
    "I’m looking for something to hold my beverage.",
    "I want something lightweight to drink from.",
    "I need something disposable to drink out of.",
    "Where can I find something to hold my soda?",
    "I need something to scoop up some liquid.",
    "I’m looking for a small, lightweight drinking vessel.",
    "I’ve got a drink, but I need something to pour it into.",
    "Something lightweight to hold my water would be useful.",
    "Where’s something I can use to hold my juice?",
    "This bottle is too big—I need something smaller to drink from.",
    "A simple container for my drink would be helpful.",
    "Is there something disposable I can use for this?",
    "I don’t want to drink straight from the bottle—what can I use?",
    "Pouring directly from the pitcher is messy—I need something else.",
    "I should find something I can reuse to reduce waste.",
    "I need something lightweight to hold my drink.",
    "A disposable drinking container would be useful right now.",
    "I don’t want to drink straight from the bottle—I need something else.",
    "I should grab something that can hold both cold and hot drinks.",
    "A small container for my drink would be really helpful.",
    "Carrying this drink around would be easier if I had something to hold it.",
    "Something stackable and easy to grab would be ideal for this drink.",
    "I need a simple, unbreakable container to pour my drink into.",
    "A cup that doesn’t feel heavy but does the job is what I need.",
    "I need a plastic cup to pour my drink into.",
    "A plastic cup would be useful for serving this juice.",
    "Where can I find a plastic cup? I don’t want to drink from the bottle.",
    "I should grab a plastic cup for my coffee."
    ]

smartphone_prompts = [
    "I need to look something up on the internet.",
    "Where can I check my messages?",
    "I want to make a phone call.",
    "I need to take a picture.",
    "I’m looking for a device to browse social media.",
    "I need to check my notifications.",
    "I want to play some music.",
    "I need something to search for information quickly.",
    "I need to respond to an email.",
    "Where is my portable communication device?",
    "I should check the latest news online.",
    "There’s something I need to search up quickly.",
    "Where can I check my messages?",
    "I should make a quick call.",
    "I need to take a picture of this moment.",
    "There’s a notification sound—I should check what it is.",
    "I wonder if anyone has texted me back yet.",
    "This would be a good time to play some music.",
    "I need to check my email before I forget.",
    "Let me look up the directions for where I'm going.",
    "I need a device that helps me stay connected to the world.",
    "A pocket-sized tool for browsing the web would be useful right now.",
    "I should check my calendar to see what I have planned today.",
    "This moment is worth capturing—I need a good camera.",
    "Where’s my go-to device for quick communication?",
    "A screen with access to everything I need would be perfect.",
    "I should send a quick message before I forget.",
    "There’s an app that could help me with this task.",
    "I need something that lets me access information instantly.",
    "I should check the notifications I just heard.",
    "I need my smartphone to check something online.",
    "Where’s my smartphone? I need to send a quick message.",
    "A smartphone would help me look up directions.",
    "I should grab my smartphone to take a picture."
]

strawberry_prompts = [
    "I’m craving something small and sweet with a little sour.",
    "I want a red fruit for a snack.",
    "I need a berry to add to my dessert.",
    "I’m looking for something juicy and bite-sized.",
    "I need something fresh to put in my smoothie.",
    "Where are the small, red fruits?",
    "I want something slightly tangy but sweet.",
    "I need something to add to my yogurt.",
    "I’m looking for a fruit that pairs well with chocolate.",
    "I need a healthy snack with a rich color.",
    "A small, sweet and sour treat would be perfect right now.",
    "This dessert would be better with something fruity.",
    "Something red and juicy would make a nice addition to my snack.",
    "I should grab something light and naturally sweet.",
    "This smoothie could use a fresh ingredient.",
    "A bite-sized fruit would be great.",
    "I want something that pairs well with chocolate.",
    "This breakfast could use a fresh topping.",
    "A handful of something sweet would be refreshing.",
    "A bright-colored fruit would add a nice touch.",
    "I’m looking for a fruit that’s small, sweet, and easy to eat.",
    "Something bright red and juicy would be a great addition to my snack.",
    "I need a fruit that has tiny seeds on its surface.",
    "A berry that’s both tart and sweet would be nice.",
    "This smoothie is missing a fresh, slightly tangy fruit.",
    "A fruit that doesn’t need peeling but tastes delicious would be ideal.",
    "I could go for something I can just pick up and pop into my mouth.",
    "I need something that pairs well with cream or chocolate.",
    "A juicy fruit with a soft bite is exactly what I want.",
    "I should find a fruit that grows in small clusters and has a fresh aroma.",
    "A strawberry would be a nice treat right now.",
    "I should add some strawberries to this dessert.",
    "Where can I find a strawberry? I need something fresh and sweet.",
    "A few strawberries would make this snack perfect."
]


reset_prompts = [
    "Return to your starting position.",
    "Go back to where you started.",
    "Move to the original spot.",
    "Move back to your original location.",
    "Return to the default stance.",
    "Can you reposition yourself to where you began?",
    "Reset to your standby position.",
    "Move back to the neutral state.",
    "Go back to the first position you were in.",
    "Please restore your default posture.",
    "Move to your resting position.",
    "Go back and prepare for the next task.",
    "Return to the base location.",
    "Reset yourself before we continue.",
    "Move back to your default alignment."
]