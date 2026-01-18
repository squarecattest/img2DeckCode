import discord
from discord.ext import commands
from description import get_localizations, get_localization_value
import asyncio
from tqdm import tqdm
from fastapi.responses import JSONResponse
from typing import Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from main import api_recognize

def _format_embed(ctx: discord.ApplicationContext, output: dict[str, Any]) -> discord.Embed:
    monster_cards = output["main_deck"]["Monster"]
    monster_cards_count = sum(monster_cards.values())
    spell_cards = output["main_deck"]["Spell"]
    spell_cards_count = sum(spell_cards.values())
    trap_cards = output["main_deck"]["Trap"]
    trap_cards_count = sum(trap_cards.values())
    main_deck_count = monster_cards_count + spell_cards_count + trap_cards_count
    extra_deck_cards = output["extra_deck"]
    extra_deck_count = sum(extra_deck_cards.values())

    main_deck_title = f"# Main Deck[{main_deck_count}/60]"
    monsters_title = f"## Monster Cards x{monster_cards_count}"
    spell_title = f"## Spell Cards x{spell_cards_count}"
    trap_title = f"## Trap Cards x{trap_cards_count}"
    extra_title = f"# Extra Deck[{extra_deck_count}/15]"
    monsters_content = "\n".join(
        f"{k} x{v}" for k, v in monster_cards.items()
    )
    spell_content = "\n".join(
        f"{k} x{v}" for k, v in spell_cards.items()
    )
    trap_content = "\n".join(
        f"{k} x{v}" for k, v in trap_cards.items()
    )
    extra_content = "\n".join(
        f"{k} x{v}" for k, v in extra_deck_cards.items()
    )
    content = "\n".join((
        main_deck_title,
        monsters_title,
        monsters_content,
        spell_title,
        spell_content,
        trap_title,
        trap_content,
    )) + "\n\n" + extra_title + "\n" + extra_content
    return discord.Embed(
        color=discord.Color.green(),
        title="Deck",
        author=discord.EmbedAuthor(ctx.user.name, icon_url=ctx.user.avatar.url),
        description=content
    )


class RecognizeProgresser:
    def __init__(self, ctx: discord.ApplicationContext) -> None:
        self.ctx = ctx
        self.curr = 0
        self.length = None
        self.finished = False
        self.event = asyncio.Event()

    def setup(self, length: int) -> None:
        self.length = length
        self.tqdm = tqdm(total=length, ncols=30, bar_format="{percentage:3.0f}% |{bar}|", ascii="╴█")
        self.event.set()

    def progress(self) -> None:
        self.curr += 1
        self.tqdm.update()
        self.event.set()

    @property
    def text(self) -> str:
        return tqdm.format_meter(**self.tqdm.format_dict)
    
    @property
    def embed(self) -> discord.Embed:
        return discord.Embed(
            color=discord.Color.green(),
            title="Progressing",
            description=self.text
        )

    async def trace_progress(self) -> None:
        while not self.finished:
            await self.event.wait()
            await self.ctx.edit(embed=self.embed)
            self.event.clear()


async def _recognize(image: discord.Attachment, progressor: RecognizeProgresser, other_task: asyncio.Task) -> JSONResponse | dict[str, Any]:
    response = await api_recognize(file=image, progressor=progressor)
    other_task.cancel()
    return response


class ImgrecCog(commands.Cog):
    @discord.slash_command(
        description=get_localization_value("recognize.description"), 
        description_localizations=get_localizations("recognize.description")
    )
    async def recognize(self, ctx: discord.ApplicationContext, image: discord.Option(discord.Attachment, description=get_localization_value("recognize.desc_image"), description_localizations=get_localizations("recognize.desc_image"))):
        await ctx.defer()
        progressor = RecognizeProgresser(ctx)
        async with asyncio.TaskGroup() as tg:
            t1 = tg.create_task(progressor.trace_progress())
            t2 = tg.create_task(_recognize(image, progressor, t1))
        response = t2.result()
        del image
        if isinstance(response, JSONResponse):
            embed = discord.Embed(
                color=discord.Color.red(),
                title=":no_entry: Error",
                description=response.body.decode()
            )
            return await ctx.edit(embed=embed)
        
        data = response["data"]
        # TODO: data process
        await ctx.edit(embed=_format_embed(ctx, output=data))
        

