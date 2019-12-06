#!/usr/bin/env python3

import sys
import math
import json
import string
import os
import io
import time
import subprocess
import concurrent.futures

from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps

import requests
import tesserocr

from pprint import pprint as pp


username = "AnSq00"


OCR_THREADS = 3


precrop_origin = (0, 51)
precrop_size   = (1920, 1003)

grid_origin  = (91, 185)
grid_size    = (7, 4)
cell_size    = (156, 156)
cell_spacing = (30, 30)

cell_name_offset  = (0, 92)
cell_name_size    = (156, 64)
cell_count_offset = (0, 0)
cell_count_size   = (25, 25)


last_warframe_market_request_time = 0
warframe_market_rate_limit = 3


def main(fnames):
    prime_parts = load_prime_parts()
    needed_primes = load_needed_primes()

    inventory = load_inventory(prime_parts)

    sell_orders = load_sell_orders(username, prime_parts)
    item_price_data = get_item_prices(inventory, sell_orders)

    my_items = assemble_my_items(inventory, sell_orders, prime_parts, item_price_data, needed_primes)

    with open("item_data.json", "w") as f:
        json.dump(my_items, f, indent=4)

    to_csv(my_items, "item_data.csv")


def load_prime_parts(fname="warframe_items.json"):
    print("Loading list of valid Prime parts")

    with open(fname) as f:
        items = json.load(f)

    prime_parts = {}
    for i in items:
        if "Prime" in i["name"] and i["type"] != "Extractor" and i["type"] != "Skin":
            if "components" in i:
                for comp in i["components"]:
                    if "ducats" in comp:
                        if i["name"] == "Odonata Prime" and comp["name"] != "Blueprint":
                            prime_parts["{} {} Blueprint".format(i["name"], comp["name"])] = comp["ducats"]
                        else:
                            prime_parts["{} {}".format(i["name"], comp["name"])] = comp["ducats"]

    return prime_parts


def filter_items(fname="warframe_items.json"):
    with open(fname) as f:
        items = json.load(f)

    prime_parts = {}
    for i in items:
        if "Prime" in i["name"] and i["type"] != "Extractor" and i["type"] != "Skin":
            if "components" in i:
                for comp in i["components"]:
                    if "ducats" in comp:
                        if i["name"] == "Odonata Prime" and comp["name"] != "Blueprint":
                            prime_parts["{} {} Blueprint".format(i["name"], comp["name"])] = i
                        else:
                            prime_parts["{} {}".format(i["name"], comp["name"])] = i

    with open("filtered_items.json", "w") as f:
        json.dump(prime_parts, f)


def load_inventory(prime_parts):
    print("Reading inventory from screenshots...")

    cells = cut_cells(fnames)
    save_cells(cells)

    cells_text = ocr_cells(cells)
    with open("inv.json", "w") as f:
        json.dump(cells_text, f)

    #check_answers(cells_text)

    inventory = {}
    for i in range(len(cells_text)):
        name = validate_name(cells_text[i][0], i, prime_parts)
        if name:
            inventory[name] = cells_text[i][1]

    return inventory


def cut_cells(fnames):
    cells = []

    for i,fname in enumerate(fnames):
        im = Image.open(fname)
        im = im.crop((precrop_origin[0], precrop_origin[1], precrop_origin[0]+precrop_size[0], precrop_origin[1]+precrop_size[1]))
        im = ImageOps.invert(im)
        im = im.point(recolor_quarter_circle)
        im = im.convert("L", (.250, .628, .122, 0))

        for row in range(grid_size[1]):
            for col in range(grid_size[0]):
                x = col * (cell_size[0] + cell_spacing[0]) + grid_origin[0]
                y = row * (cell_size[1] + cell_spacing[1]) + grid_origin[1]

                cell_im = im.crop((x, y, x+cell_size[0], y+cell_size[1]))

                cell_name_im  = cell_im.crop((cell_name_offset[0], cell_name_offset[1], cell_name_offset[0]+cell_name_size[0], cell_name_offset[1]+cell_name_size[1]))

                cell_count_im = cell_im.crop((cell_count_offset[0], cell_count_offset[1], cell_count_offset[0]+cell_count_size[0], cell_count_offset[1]+cell_count_size[1]))
                cell_count_im = Image.eval(cell_count_im, lambda x: 0 if x < 100 else 255)

                cell_num = i * grid_size[0] * grid_size[1] + row * grid_size[0] + col
                cells.append((cell_im, cell_name_im, cell_count_im, cell_num))

    return cells


def recolor_quarter_circle(x):
    return round(math.sqrt(255**2 - (x - 255)**2))


def save_cells(cells):
    mkdir_if_not_exists("cells")
    for i, cell in enumerate(cells):
        cell[0].save("cells/cell_{}.png".format(i))
        cell[1].save("cells/cell_{}_name.png".format(i))
        cell[2].save("cells/cell_{}_count.png".format(i))


def mkdir_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def ocr_cells(cells):
    ex = concurrent.futures.ThreadPoolExecutor(OCR_THREADS)
    results = ex.map(ocr_cell, cells)
    return list(results)


def ocr_cell(cell):
    cell_name = ocr(cell[1], string.ascii_letters+" &").strip().replace("\n", " ")

    cell_count = ocr(cell[2], string.digits, True).strip()
    cell_count = int(cell_count) if cell_count else 1

    print("\t[{}]\t\t{}\t{}".format(cell[3], cell_count, cell_name))
    return (cell_name, cell_count)


def ocr_subprocess(im, alphabet=None, single_line=False):
    with io.BytesIO() as im_file:
        im.save(im_file, "PNG")

        cmdline = ["tesseract", "stdin", "stdout", "--tessdata-dir", "tessdata"]
        if alphabet:
            cmdline += ["-c", "tessedit_char_whitelist={}".format(alphabet)]
        if single_line:
            cmdline += ["--psm", "7"]

        p = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        text, _ = p.communicate(im_file.getvalue())
    return text.decode()


def ocr_tesserocr(im, alphabet=None, single_line=False):
    with tesserocr.PyTessBaseAPI() as tess:
        if single_line:
            tess.SetPageSegMode(tesserocr.PSM.SINGLE_LINE)
        if alphabet:
            tess.SetVariable("tessedit_char_whitelist", alphabet)

        tess.SetImage(im)
        result = tess.GetUTF8Text()
    return result


ocr = ocr_subprocess
#ocr = ocr_tesserocr


def check_answers(cells_text, answers_fname="test_data/answers.json"):
    with open(answers_fname) as f:
        answers = json.load(f)

    name_fails = 0
    count_fails = 0
    for i in range(len(answers)):
        if cells_text[i][0] == answers[i][0]:
            if cells_text[i][1] == answers[i][1]:
                print('  \t{}\t"{}"'.format(cells_text[i][1], cells_text[i][0]))
            else:
                print('count\t{}->{}\t"{}"'.format(cells_text[i][1], answers[i][1], cells_text[i][0]))
                count_fails += 1
        else:
            if cells_text[i][1] == answers[i][1]:
                print('name\t{}\t"{}" -> "{}"'.format(cells_text[i][1], cells_text[i][0], answers[i][0]))
                name_fails += 1
            else:
                print('both\t{}->{}\t"{}" -> "{}"'.format(cells_text[i][1], answers[i][1], cells_text[i][0], answers[i][0]))
                name_fails += 1
                count_fails += 1
    print("\nFailures:\n\tname:  {1}/{0}\n\tcount: {2}/{0}\n\ttotal: {3}/{4}".format(len(answers), name_fails, count_fails, name_fails+count_fails, 2*len(answers)))


def validate_name(name, cell_num, prime_parts):
    if not name:
        return ""

    split = name.split()

    if "Prime" not in split:
        return validate_name(input('"{}" (cell {}) does not appear to be a valid Prime part. What is it supposed to be? (Blank to discard): '.format(name, cell_num)), cell_num, prime_parts)

    base_item = split[:split.index("Prime")]
    part = split[split.index("Prime") + 1:]

    # Odonata (the only Prime Archwing) has the Archwing symbol in front of its blueprint name, which the OCR picks up wrong
    if "Odonata" in base_item:
        base_item = ["Odonata"]

    # The OCR sometimes misreads "Systems"
    if "Blueprint" in part and len(part) >= 2 and part[0] not in ("Neuroptics", "Chassis", "Systems", "Wings", "Harness") and "y" in part[0]:
        print('Correcting "{}" in "{}" to "Systems"'.format(part[0], name))
        part[0] = "Systems"

    corrected_name = " ".join(base_item + ["Prime"] + part)
    if corrected_name in prime_parts:
        return corrected_name
    else:
        return validate_name(input('"{}" (cell {}) does not appear to be a valid Prime part. What is it supposed to be? (Blank to discard): '.format(corrected_name, cell_num)), cell_num, prime_parts)


def load_sell_orders(username, prime_parts):
    print("Loading sell orders for {}".format(username))

    orders = json.loads(requests.get("https://api.warframe.market/v1/profile/{}/orders".format(username)).text)["payload"]["sell_orders"]

    items = {}
    for order in orders:
        name = order["item"]["en"]["item_name"]

        # Warframe parts
        name = name.replace("Neuroptics", "Neuroptics Blueprint")
        name = name.replace("Chassis",    "Chassis Blueprint")

        # Archwing parts
        name = name.replace("Harness", "Harness Blueprint")
        name = name.replace("Wings",   "Wings Blueprint")

        # Systems can be a Warframe, Archwing, or Sentinel part, but Sentinel parts don't have blueprints
        bp_name = name.replace("Systems", "Systems Blueprint")
        if bp_name in prime_parts:
            name = bp_name

        items[name] = {"quantity": order["quantity"], "platinum": int(order["platinum"])}

    return items


def find_quantity_differences(inventory, sell_orders, include_noninventory=True, include_nonmarket=True):
    quant_diffs = {}
    for item in (inventory.keys() | sell_orders.keys()):
        inv_count = inventory[item] if item in inventory else 0
        sell_count = sell_orders[item]["quantity"] if item in sell_orders else 0

        if (not include_noninventory and not inv_count) or (not include_nonmarket and not sell_count):
            continue

        if inv_count != sell_count:
            quant_diffs[item] = {"inventory": inv_count, "market": sell_count}

    return quant_diffs


def get_item_prices(inventory, sell_orders):
    print("Getting item price data...")

    item_price_data = {}

    ex = concurrent.futures.ThreadPoolExecutor(4)
    results = ex.map(get_item_price_data, sorted(list(inventory.keys() | sell_orders.keys())))
    for r in results:
        item_price_data[r[0]] = r[1]

    return item_price_data


def get_item_price_data(item):
    price_data = {
        "live_orders" : [],
        "volume" : [],
        "min"    : [],
        "max"    : [],
        "mean"   : [],
        "median" : [],
        "moving" : [],
        "vwap"   : []
    }

    market_url_name = item.lower().replace(" ", "_").replace("&", "and")
    market_url_name = market_url_name.replace("neuroptics_blueprint", "neuroptics")
    market_url_name = market_url_name.replace("chassis_blueprint",    "chassis")
    market_url_name = market_url_name.replace("systems_blueprint",    "systems")
    market_url_name = market_url_name.replace("wings_blueprint",      "wings")
    market_url_name = market_url_name.replace("harness_blueprint",    "harness")

    if "imprint" not in market_url_name:
        market_url_name = market_url_name.replace("kubrow_", "")

    orders = warframe_market_request("https://api.warframe.market/v1/items/{}/orders".format(market_url_name), lambda x: x["payload"]["orders"])
    for o in orders:
        if o["order_type"] != "sell" \
            or o["platform"] != "pc" \
            or o["region"] != "en" \
            or not o["visible"] \
            or o["user"]["status"] != "ingame":
                continue

        price_data["live_orders"].append(int(o["platinum"]))
    price_data["live_orders"].sort()

    seven_day_stats = warframe_market_request("https://api.warframe.market/v1/items/{}/statistics".format(market_url_name), lambda x: x["payload"]["statistics_closed"]["90days"][-7:])
    for day in seven_day_stats:
        for stat in (
                ('volume',     'volume', int),
                ('min_price',  'min',    int),
                ('max_price',  'max',    int),
                ('avg_price',  'mean',   float),
                ('median',     'median', float),
                ('moving_avg', 'moving', float),
                ('wa_price',   'vwap',   float)
            ):
            try:
                price_data[stat[1]].append(stat[2](day[stat[0]]))
            except KeyError:
                price_data[stat[1]].append(None)

    print('\t{}\t{}'.format(price_data["median"][-1], item))
    return (item, price_data)


def warframe_market_request(url, extractor):
    global last_warframe_market_request_time

    time_now = time.time()
    if last_warframe_market_request_time + 1/warframe_market_rate_limit > time_now:
        time.sleep(last_warframe_market_request_time + 1/warframe_market_rate_limit - time_now)

    last_warframe_market_request_time = time.time()
    r = requests.get(url)
    try:
        return extractor(json.loads(r.text))
    except:
        print(url)
        print(r.text)
        raise


def load_needed_primes(fname="primes_needed.txt"):
    needed_primes = []
    with open(fname) as f:
        for line in f:
            needed_primes.append(line.strip())
    return tuple(needed_primes)


def assemble_my_items(inventory, sell_orders, prime_parts, item_price_data, needed_primes):
    my_items = {}
    for item in sorted(list(inventory.keys() | sell_orders.keys())):
        item_data = {}

        item_data["needed"] = item.split("Prime")[0].startswith(needed_primes)
        item_data["inventory"] = inventory[item] if item in inventory else 0
        item_data["my_sell_order"] = sell_orders[item] if item in sell_orders else {"quantity": 0, "platinum": None}
        item_data["price_data"] = item_price_data[item]
        item_data["platinum"] = item_data["price_data"]["median"][-1]
        item_data["ducats"] = prime_parts[item] if item in prime_parts else None

        try:
            item_data["ducats_per_plat"] = item_data["ducats"] / item_data["platinum"]
        except (TypeError, ZeroDivisionError):
            item_data["ducats_per_plat"] = None

        try:
            item_data["plat_per_ducat"] = item_data["platinum"] / item_data["ducats"]
        except (TypeError, ZeroDivisionError):
            item_data["plat_per_ducat"] = None

        my_items[item] = item_data

    return my_items


def to_csv(my_items, fname="item_data.csv"):
    print("Writing CSV file")

    with open(fname, "w") as f:
        f.write('Item,Needed,Inventory,Sell Order Quantity,Quantity Diff,Sell Order Platinum,Platinum,Ducats,Ducats per Plat,Plat per Ducat,Num Live Orders,Live Orders\n')
        for item in sorted(list(my_items.keys())):
            i = my_items[item]
            f.write("{},".format(item))
            f.write("{},".format("Yes" if i["needed"] else ""))
            f.write("{},".format(i["inventory"] or ""))
            f.write("{},".format(i["my_sell_order"]["quantity"] or ""))
            f.write("{},".format((i["inventory"] - i["my_sell_order"]["quantity"] if i["inventory"] and i["my_sell_order"]["quantity"] else "") or ""))
            f.write("{},".format(i["my_sell_order"]["platinum"] or ""))
            f.write("{},".format(i["platinum"]))
            f.write("{},".format(i["ducats"] or ""))
            f.write("{:.2f},".format(i["ducats_per_plat"]) if i["ducats_per_plat"] else ",")
            f.write("{:.2f},".format(i["plat_per_ducat"]) if i["plat_per_ducat"] else ",")
            f.write("{},".format(len(i["price_data"]["live_orders"])))
            f.write("{}\n".format(" ".join(str(x) for x in i["price_data"]["live_orders"][:10]) + (" ..." if len(i["price_data"]["live_orders"]) > 10 else "")))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--x":
        with open("item_data.json") as f:
            item_data = json.load(f)
        to_csv(item_data)
        sys.exit()

    fnames = sys.argv[1:]
    main(fnames)
