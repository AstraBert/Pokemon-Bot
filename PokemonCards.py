from datasets import load_dataset
import random as r

dataset = load_dataset("TheFusion21/PokemonCards")
image_urls = dataset["train"]["image_url"]
image_des = dataset["train"]["caption"]
image_name = dataset["train"]["name"]
image_set_name = dataset["train"]["set_name"]

def choose_random_cards():
    global image_urls, image_des, image_name, image_set_name
    indexes = []
    n = 0
    while n < 5:
        l = r.randint(0,len(image_urls))
        if l not in indexes:
            n+=1
            indexes.append(l)
        else:
            continue
    basestr = ""
    c = 0
    for idx in indexes:
        c+=1
        llmstr = f"CARD {c}:\n\nNAME:\n{image_name[idx]}\n\SET_NAME:\n{image_set_name[idx]}\n\nDESCRIPTION:\n{image_des[idx]}"
        basestr+=llmstr+"\n\n\n"
    urls = [image_urls[i] for i in indexes]
    return basestr, urls
