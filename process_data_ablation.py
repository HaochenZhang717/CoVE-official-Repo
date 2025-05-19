import csv
import json
import pandas as pd
from tqdm import tqdm
import re
import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="none", help="data root path")
args = parser.parse_args()
# root = "dataset/amazon/raw/beauty"
id_title = {}
with open(f"{args.root}/meta.json", "r") as f:
    metadata = f.read()

for line in tqdm(metadata.strip().split("\n")):
    line = ast.literal_eval(line)
    if 'title' in line.keys():
        if len(line['title']) > 1:  # remove the item without title
            id_title[line['asin']] = line['title']

with open(f"{args.root}/datamaps.json") as f:
    # datamaps = [json.loads(line) for line in f]
    datamaps = json.load(f)
# item2id = datamaps["item2id"]

id2title = {}


missing_asin_title_map = {
 'B00CMQTVUA': "Xbox One + Kinect",
 'B003B7Q5YY': "Rhode Island Novelty Easter Eggs Bght Plastic Egg Assortment 144 Pieces",
 'B0038AR4VM': "LeapFrog Globe: Earth Adventures Learning Game (works with LeapPad Tablets & LeapsterGS)",
 'B002I0K3CK': "New Super Mario Bros. U",
 'B00BT2BFGG': "no title",
 'B00009VDXX': "Jakks / Namco Arcade Classics Plug and Play TV Games",
 'B001KMRN0M': "PlayStation Portable 3000 Core Pack System - Piano Black",
 'B00ECOBFA4': "The LEGO Movie Videogame - Xbox 360 Standard Edition",
 'B004FSE52C': "Tomb Raider",
 'B003O6EB70': "BioShock Infinite - Xbox 360",
 'B002R7IK74': "no title",
 'B000FQ2DTA': "Final Fantasy XIII - Playstation 3",
 'B0050D1V2I': "Play-Doh 36 Can Mega Pack of Non-Toxic Modeling Compound, 3 Oz Cans",
 'B009TQA8OY': "Super Soaker Nerf Hydro Pack",
 'B003BMJKI2': "Super Scribblenauts",
 'B0050SX00Y': "LittleBigPlanet Karting",
 'B00936K09S': "Xbox 360 Wireless Controller with Transforming D-Pad and Play and Charge Kit - Black",
 'B007SRM5U6': "Batman: Arkham City - Game of the Year Edition (Restricted distribution)",
 'B006KYYOSO': "Tekken Tag Tournament 2 PS3",
 'B0088I7L76': "Injustice: Gods Among Us - Playstation 3",
 'B004OCK9KG': "Sesame Street: Once Upon A Monster - Xbox 360",
 'B00BU3ZLJQ': "Minecraft – Xbox 361",
 'B0033BM3K8': "no title",
 'B006ZMGR5E': "Learning Resources Candy Construction Building Set - 92 Pieces, Ages 4+,Toddler Learning Toys, Fine Motor Building Toy, Preschool Toys, STEM Toys",
 'B008CP6KG8': "Scribblenauts Unlimited - Nintendo 3DS",
 'B000062SQK': "no title",
 'B002BS4JDS': "LEGO Harry Potter: Years 1-4 - Xbox 360",
 'B00GZ1GUSY': "Tomb Raider: Definitive Edition - PlayStation 4",
 'B001ESQTUM': "Pokémon Rare Grab Bag 20 Rare Pokémon Cards",
 'B00AXI9WNU': "no title",
 'B00FM5IY38': "Ryse: Son of Rome XBOX one",
 'B00342H7ZW': "no title",
 'B00DUARDFC': "Lego Marvel Super Heroes",
 'B003ET17HY': "no title",
 'B00HGLLRV2': "inFAMOUS: Second Son Standard Edition (PlayStation 4)",
 'B00CMC6HZ6': "WWE 2K14 - Xbox 360",
 'B001A5SNXA': "Toysmith Jumbo Sidewalk Chalk, Assorted Colors (Packaging May Vary), For Boys & Girls Ages 3+",
 'B001QGUFDO': "On the Edge 900124 Red Folding Utility Wagon With Handle",
 'B0026NDRWW': "Melissa Doug Sunny Patch Augie Alligator Sleeping Bag",
 'B00B98HG18': "LEGO: Marvel Super Heroes - Nintendo Wii U",
 'B00BGAA3S2': "PlayStation 4 Camera (Old Model)",
 'B00BGA9YZK': "Killzone: Shadow Fall (PlayStation 4)",
 'B00AXE639A': "no title",
 'B00A0GNOVQ': "no title",
 'B00BUDJRD2': "no title",
 'B00CH9253W': "Mario Party 3DS",
 'B001UCIM8E': "ALEX Toys Little Hands Wash & String",
 'B00DUARBTA': "LEGO Marvel Super Heroes - PlayStation 4",
 'B00CW3E9NM': "Scribblenauts Unmasked - A DC Comics Adventure - Nintendo 3DS",
 'B0051TLAZ4': "LEGO Harry Potter: Years 5-7 - Playstation 3",
 'B0013WLX6O': "no title",
 'B00ECOAX34': "The LEGO Movie Videogame - Wii U",
 'B00002DHEV': "Nintendo 64 System - Video Game Console",
 'B00FM5IY4W': "Forza Motorsport 5",
 'B00FJWNSU8': "Injustice: Gods Among Us - Ultimate Edition",
 'B0050SVTUM': "no title",
 'B000P51LDK': "Toysmith Bug Out Bob Toy",
 'B00DWXUYN0': "HORI Retro Mario Hard Pouch for NEW 3DS XL and Nintendo 3DS XL",
 'B0028CY8M4': "no title",
 'B003HC8NUW': "HTC DROID Incredible, Black (Verizon Wireless)",
 'B007M6W38W': "NBA 2K13 - Xbox 360",
 'B007H2PXLK': "Camco 51800 Director's Chair (Black Swirl)",
 'B000AL2RI2': "Permatex 81150 Dielectric Tune-Up Grease, 0.33 oz. Tube, Silver",
 'B002I0H9WM': "Your Shape Fitness Evolved - Xbox 360",
 'B003N9B3CY': "no title",
 'B0000AY1W1': "303 30370 Marine and Recreation Aerospace Protectant - 1 Gallon",
 'B0002UEPVI': "Permatex 80050 Clear RTV Silicone Adhesive Sealant, 3 oz",
 'B001MBUGLY': "EA Sports Active",
 'B000EBUG4K': "no title",
 'B002PTNXZI': "Crosman PL177 .177-Caliber 16-Round Pellet Loader",
 'B001VFVTOY': "Troy Industries Two Point Battle Sling",
 'B002UUTCKC': "Motorola DROID A855 Android Phone (Verizon Wireless)",
 'B00CF7KVMI': "Madden NFL 25 Anniversary Edition with NFL Sunday Ticket -Xbox 360",
 'B000I4JPZE': "Camco Tandem Tire Changing Ramp with 4.5-Inch Lift, Yellow (21000)",
 'B0014E3MSS': "Camco Pigtail Propane Hose Connector | Designed to Connect to an RV or Trailer Propane Regulator | Safety Features Include Thermal Protection and Excess Flow Protection | 12-inch (59053)",
 'B0002UEN1U': "Permatex 82180 Ultra Black Maximum Oil Resistance RTV Silicone Gasket Maker, Sensor Safe And Non-Corrosive, For High Flex And Oil Resistant Applications 3 oz",
 'B000GPV2QA': "no title",
 'B004XDDOPI': "Casio G'zOne Commando Android Phone (Verizon Wireless)",
 'B000BUTD8Y': "Camco 40542 Heavy-Duty 20lb or 30lb Dual Propane Tank Cover (Polar White)",
 'B005C5QLVK': "no title",
 'B000H3BLBM': "Victorinox Swiss Army One-Hand Trekker Multi-Tool Pocket Knife",
 'B000SKY3GO': "Tri-Flow TF20006 Superior Lubricant, 12-Ounce Aerosol",
 'B0000AXNJS': "Flitz Metal Polish and Cleaner Liquid for All Metal, Also Works On Plastic, Fiberglass, Aluminum, Jewelry, Sterling Silver: Great for Headlight Restoration and Rust Remover, 3.4 oz",
 'B002SIR91A': "Lifeline Aluminum Sport Utility Shovel, 3 Piece Collapsible Design, Perfect Snow Shovel for Car, Camping and Other Outdoor Activities, Red",
 'B00029WY9Y': "K&N Air Filter Oil: 6.5 Oz Aerosol; Restore Engine Air Filter Performance and Efficiency, 99-0504",
 'B00449JT82': "ChampionShell Pouch",
 'B0000AY1W7': "303 Marine Fabric Guard - Restores Water and Stain Repellency To Factory New Levels, Simple and Easy To Use, Manufacturer Recommended, Safe For All Fabrics, 1 Gallon (30674), white",
 'B0029U0CUW': "no title",
 'B004N18DFG': "REAL AVID GUN TOOL",
 'B0025KZV84': "NHL 10 - Xbox 360",
 'B0000AY9W6': "Camco Standard Wheel Chock for Trailers & RVs - Heavy Duty Trailer Wheel Chock Constructed of Durable Plastic w/UV inhibitors - for Use w/Tires Up to 26-inches - Re-Hitch in Confidence - 44412",
 'B005UE413I': "Kershaw One Ton Folding Knife",
 'B001DZT0M0': "Victorinox Swiss Army One-Hand Trekker Multi-Tool Pocket Knife",
 'B000FNDVGC': "Victorinox Swiss Army Manager Pocket Knife",
 'B002YT2COM': "GSI Outdoors Fairshare Mug, 22 Ounce Collapsible Mug for Camping, Trail, Travel or Office - Green",
 'B002KJ9XE8': "EA Sports Active: More Workouts - Nintendo Wii",
 'B000F7T554': "Remington T-10 True Junior Shooting Glasses",
 'B00386VMWI': "NCAA Football 11 - Playstation 3",
 'B0061OQJTK': "no title",
 'B004ZDL2WI': "Topeak Survival Tool Wedge II Bike Pack with Fixer 25",
 'B00A750QIE': "MLB 13 The Show",
 'B000IQ9PMA': "Planet Bike Phil Hose CO2 Tire Pump",
 'B000XTPO2M': "Master Lock 8291DPS 3' Black HardenedSteel Chain with Integrated Keyed Lock",
 'B005OCYTS8': "no title",
 'B00009PGNS': "Koolatron 12V Electric Car Cooler/Warmer 17L (18 qt) - Portable Thermoelectric Fridge for Camping Travel Road Trips and Picnics - Lightweight Soft-Sided and Compact",
 'B007BGRASQ': "Pachmayr Tactical Grip Glove for Glock 26, 27, 28, 29, 30, 33, 36, 39",
 'B000AAM2Z6': "Permatex 27100 High Strength Threadlocker Red, 6 ml",
 'B002IKWG4I': "Decade Motorsport Street Rocket Black and Gray Gloves",
 'B0000AMK8R': "no title",
 'B007M6W3A0': "NBA 2K13 - Playstation 3",
 'B00AY1CT4U': "Madden NFL 25 - Xbox 360",
 'B00AY1CT3G': "Madden NFL 25 - Playstation 3",
 'B0002ITTK2': "Tri-Flow TF21010 Superior Lubricant with Drip Bottle- 2 oz , Black/Red",
 'B00AY1ALVS': "NCAA Football 14 - Playstation 3",
 'B000H7CKAY': "STA-BIL (22001) Fogging Oil - Stops Corrosion In Stored Engines - Lubricates And Protects Cylinders - Coats Internal Engine Components - For All 2 and 4 Cycle Engines, 12 oz.",
 'B000OF5M04': "Peakfun Dry PAK Dry Bag Case for Cell Phones, iPhone, Androids, 5\" x 8\"",
 'B00C8YXGB6': "Wagan EL2639 FRED Flashing Roadside Emergency Disc LED Flare",
 'B000PD6R8G': "Hornady Lock N Load Shell Plate – for Reliable Caliber Changes on Your Lock-N-Load Press – Each Plate Works with Multiple Calibers – Easy Case Insertion",
 'B000LW1TRU': "Zanheadgear® Full Mask Neoprene Black",
 'B00BQS5B7W': "Weapon",
 'B003AHLX1U': "Sunday Afternoons Sun Tripper Cap",
 'B001U34NYA': "Hammers Wholesale Lot 48pcs Genuine 20mm Small Mini Compasses for Survival Kits",
 'B002IKWGV6': "Decade Motorsport Street Classic Black Gloves",
 'B004XNQLB2': "Camco 42613 3\" x 15\' Awning Repair Tape",
 'B004YW5L36': "VOODOO TACTICAL Men's Deluxe Professional Special Ops Field Medical Pack, Coyote, Large",
 'B0000DZV3K': "Coghlan's Folding Travel Scissors",
 'B004DI5H26': "MLB 11: The Show",
 'B001RMWO9A': "Pro Ears Pro Mag Gold, Electronic Hearing Protection, Amplification, Shooting, Exclusive DLSC Compression, Made in USA",
 'B00AY1ALTU': "NCAA Football 14 - Xbox 360",
 'B005K4TWVY': "HORI PS3 Tactical Assault Commander 3 (T.A.C.) for FPS Games (Camouflage Version)",
 'B001JJCHN4': "Barska Huntmaster Crosshair Reticle Rifle Scope for Hunting & Target Shooting",
 'B00CF7KVGE': "Madden NFL 25 Anniversary Edition with NFL Sunday Ticket - Playstation 3",
 'B001R33Y5W': "Duty Two Point Sling (Strap alone)",
 'B0012TRLWS': "ZANheadgear Neoprene Full Face Mask",
 'B004FGX2H8': "VFG1, V-Grip Mounting Rack for Gun, Bow, Tools, Utilities - Single Rider",
 'B000K7IJGA': "Oakley Mx Accessory Lens, Dark Grey, One Size",
 'B003QSAZN0': "MysticalBlades New Rose Ebony Machette W Wooden Handle Machete Sheath",
 'B004C444R0': "Milwaukee Motorcycle Clothing Company MMCC Riding Gloves with Gel Palm (Black, Large)",
 'B00127PZCS': "MLB 08 The Show - Playstation 3",
 'B001BZK5EE': "LG Dare, Black (Verizon Wireless)",
 'B002HMD8W6': "Oakley O-FRAME MX Goggle",
 'B005JK5HQI': "no title",
 'B001V8UKBO': "\"Plus\" Tandem Tire Changing Ramp, The Fast and Easy Way To Change A Trailer's Flat Tire, Holds up to 15,000 Pounds, 5.5 Inch Lift (Yellow) (23) (21002)",
 'B00192CSI0': "LG enV2, Black (Verizon Wireless)",
 'B0044TXGU4': "Heavenly Acupressure Mats & Pillow Combo | Back & Neck Pain Relief Treatment - (Blue)",
 'B005KXHY4W': "Rage Elbow Black Soft Shell Pad",
 'B000A1329U': "Motorola RAZR V3 Black Phone (AT&T)",
 'B003400CP6': "Condor Pistol Case, Small, Black",
 'B004CDGG4U': "Major League Baseball 2K11 - Xbox 360",
 'B001DJ2USM': "L'Oreal Paris Elnett Satin Extra Strong Hold Hairspray 11 Ounce (1 Count) (Packaging May Vary)",
 'B00ALHISAQ': "BaBylissPRO Nano Titanium Rotating Hot Air Brush, Blue, 2 Inch (Pack of 1)",
 'B001MP0T2Q': "DL Professional French Manicure Clean-Up Brush, 4 Ounce",
 'B001S3I57S': "no title",
 'B00B4C35IW': "ACURE Seriously Soothing Cleansing Cream | 100% Vegan | For Dry to Sensitive Skin | Peony Extract & Chamomille - Soothes , Hydrates & Cleanses | 4 Fl Oz",
 'B000GZES7A': "Mary Kay Timewise Firming Eye Cream,0.5 oz",
 'B001RD37ZO': "OPI Nail Lacquer, Miami Beet, Purple Nail Polish, 0.5 fl oz"
}


# missing_asin = []
for asin, id in datamaps["item2id"].items():
    if asin in id_title.keys():
        title = id_title[asin]
        id2title.update({id: title})
    else:
        # missing_asin.append(asin)
        assert asin in missing_asin_title_map.keys()
        id2title.update({id: missing_asin_title_map[asin]})
        # print(f"missing title for asin {asin}, id {id}")


interactions = {}
with open(f'{args.root}/sequential_data.txt', "r") as file:
    for line in file:
        numbers = list(map(int, line.strip().split()))  # Convert each line into a list of integers
        key = 'user' + str(numbers[0])  # First number as the key
        values = numbers[1:]  # Remaining numbers as values
        interactions[key] = values  # Store in a dictionary


json_list_train = []
json_list_eval = []
json_list_eval_input_only = []
json_list_eval_output_only = []

for user, interaction in tqdm(interactions.items()):
    interaction_titles = [id2title[str(i)] for i in interaction]
    # construct train datum
    train_input = interaction[:-3]
    train_output = interaction[-3]
    history_train = "Here is the list of items that the user has bought: "
    for i in range(len(train_input)):
        if i == 0:
            history_train += "<|" + str(train_input[i]) + "|>"
        else:
            history_train += ", <|" + str(train_input[i]) + "|>"
    train_target = "<|" + str(train_output) + "|>"
    json_list_train.append({
                "text": f" Instruction: Given a list of items the user has bought before, please recommend a new item that the user likes to the user. \n {history_train}, {train_target}",
            })

    # construct eval datum
    eval_input = interaction[:-2]
    eval_output = interaction[-2]
    history_eval = "Here is the list of items that the user has bought: "
    for i in range(len(eval_input)):
        if i == 0:
            history_eval += "<|" + str(eval_input[i]) + "|>"
        else:
            history_eval += ", <|" + str(eval_input[i]) + "|>"
    eval_target = "<|" + str(eval_output) + "|>"
    json_list_eval.append({
        "text": f" Instruction: Given a list of items the user has bought before, please recommend a new item that the user likes to the user. \n {history_eval}, {eval_target}",
    })

    json_list_eval_input_only.append({
        "text": f" Instruction: Given a list of items the user has bought before, please recommend a new item that the user likes to the user. \n {history_eval}, ",
    })

    json_list_eval_output_only.append({
        "text": "<|" + str(eval_output) + "|>",
    })


    with open(f"{args.root}/train_no_text.json", 'w') as f:
        json.dump(json_list_train, f, indent=4)

    with open(f"{args.root}/test_no_text.json", 'w') as f:
        json.dump(json_list_eval, f, indent=4)

    with open(f"{args.root}/test_input_only_no_text.json", 'w') as f:
        json.dump(json_list_eval_input_only, f, indent=4)

    with open(f"{args.root}/test_output_only_no_text.json", 'w') as f:
        json.dump(json_list_eval_output_only, f, indent=4)




