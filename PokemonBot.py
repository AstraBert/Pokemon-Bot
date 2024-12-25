from discord import Client, Intents
from dotenv import load_dotenv
import os
import time
import random as r
from ChatMemory import PGClient, ConversationHistory
from ChatCohere import chat_completion, summarize
from PokemonCards import choose_random_cards
from QdrantRag import NeuralSearcher, SemanticCache, qdrant_client, encoder, image_encoder, processor, sparse_encoder
load_dotenv()

CHANNEL_ID = int(os.getenv("channel_id"))
TOKEN = os.getenv("discord_bot")
pg_db = os.getenv("pgql_db")
pg_user = os.getenv("pgql_user")
pg_psw = os.getenv("pgql_psw")

pg_conn_str = f"postgresql://{pg_user}:{pg_psw}@localhost:5432/{pg_db}"
pg_client = PGClient(pg_conn_str)

searcher = NeuralSearcher("pokemon_texts", "pokemon_images", qdrant_client, encoder, image_encoder, processor, sparse_encoder)
semantic_cache = SemanticCache(qdrant_client, encoder, "semantic_cache", 0.75)

usr_id = r.randint(1,10000)
convo_hist = ConversationHistory(pg_client, usr_id)
convo_hist.add_message(role="system", content="You are a Pokemon expert. You know everything about their characteristics, abilities, and evolutions. You always reply with the most accurate information, you are polite and helpful. Your output cannot be larger than 1500 characters (spaces included).")

intents = Intents.default()
intents.messages = True

bot = Client(intents=intents)

@bot.event
async def on_ready():
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        # Print a confirmation
        print(f"Connected to the channel: {channel.name} ({channel.id})")

        # Send information about the functioning of the bot
        await channel.send(
            f"## PokemonBot\n\nHi!😊\n\nI'm PokemonBot, your personal Pokemon expert and assistant. You can add me as an app to your Discord and chat with me at [this link](https://discordapp.com/oauth2/authorize?client_id=1320103069689708736), otherwise you can click on my name and open a direct chat with me in your private messages💬\n\nYou can use me in the following ways:\n\n- **Direct message**: if you send me a direct message, I will produce a response in 15 to 60s\n- **Command !whatpokemon**: If you put the command !whatpokemon, you should attach an image of a Pokemon and you will get out the name of it\n- **Command !cardpackage**: If you input the !cardpackage command, you will get out 5 Pokemon cards with their description\n\nHave fun!🍕"
        )
    else:
        print(
            "Unable to find the specified channel ID. Make sure the ID is correct and the bot has the necessary permissions."
        )

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    elif message.content: 
        if not message.content.startswith("!"):
            print(
                f"Got content {message.content} from {message.author}"
            )
            answer = semantic_cache.search_cache(message.content)
            if answer != "":
                await message.channel.send(answer)
            else:
                context_search = searcher.search_text(message.content)
                reranked_context = searcher.reranking(message.content, context_search)
                context = "\n\n-----------------\n\n".join(reranked_context)
                convo_hist.add_message(role="user", content=message.content)
                convo_hist.add_message(role="assistant", content="Context:\n\n"+context)
                response = chat_completion(convo_hist.get_conversation_history())
                convo_hist.add_message(role="assistant", content=response)
                semantic_cache.upload_to_cache(message.content, response)
                await message.channel.send(response)
        else:
            if message.content.startswith("!whatpokemon"):
                if message.attachments:
                    # Get the first attachment
                    attachment = message.attachments[0]
                    
                    # Create a directory for the images if it doesn't exist
                    if not os.path.exists('pokemon_images'):
                        os.makedirs('pokemon_images')
                    
                    # Generate a unique filename using timestamp
                    timestamp = int(time.time())
                    file_extension = attachment.filename.split('.')[-1]
                    filename = f'pokemon_{timestamp}.{file_extension}'
                    save_path = os.path.join('pokemon_images', filename)
                    
                    # Download the image
                    await attachment.save(save_path)
                    
                    result = searcher.search_image(save_path)
                    results = "\n".join(result)
                    await message.channel.send("You Pokemon might be:\n" + results)
                else:
                    await message.channel.send("You need to attach an image of a Pokemon to use this command")   
            elif message.content.startswith("!cardpackage"):
                description, cards = choose_random_cards()
                package = [f"![Card {i+1}]({cards[i]})" for i in range(len(cards))]
                cards_message = "\n\n".join(package)
                natural_lang_description = chat_completion([{"role": "system", "content": "You are an expert in Pokemon cards. You know everything about their characteristics, abilities, and evolutions. You always reply with the most accurate information, you are polite and helpful."}, {"role": "user", "content": f"Can you enthusiastically describe the cards in this package?\n\n{description}"}])
                await message.channel.send("## Your package:\n\n" + cards_message + "\n\n## Description:\n\n" + summarize(natural_lang_description)) 
            else:
                await message.channel.send("The command you provided through your message is not recognized, please try again.") 
bot.run(TOKEN)

## INSTALL PSYCOPG2!!!

