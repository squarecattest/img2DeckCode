import discord
from bot import ImgrecCog

bot = discord.Bot()
bot.add_cog(ImgrecCog(bot))
@bot.event
async def on_ready():
    string = f"Logged in as {bot.user} (ID: {bot.user.id})"
    print(string)
    print("-" * len(string))

bot.run("<token>")