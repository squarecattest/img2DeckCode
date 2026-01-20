import discord
from discord.ext import commands
from description import get_localizations, get_localization_value
from constants import *
import asyncio
from tqdm import tqdm
from fastapi.responses import JSONResponse
from typing import Any
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from main import api_recognize


try:
    from src.translator import translate
except ImportError:
    def translate(output: dict[str, Any], lang: str = "en-US") -> dict[str, Any]:
        '''
        Translate informations (e.g. card names) into specified language.
        '''
        return output


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


class ImgrecCog(commands.Cog):
    @staticmethod
    def _recognize_output(ctx: discord.ApplicationContext, data: dict[str, Any]) -> discord.Embed:
        data = translate(data, ctx.locale)
        monster_cards = data["main_deck"]["Monster"]
        monster_cards_count = sum(monster_cards.values())
        spell_cards = data["main_deck"]["Spell"]
        spell_cards_count = sum(spell_cards.values())
        trap_cards = data["main_deck"]["Trap"]
        trap_cards_count = sum(trap_cards.values())
        main_deck_count = monster_cards_count + spell_cards_count + trap_cards_count
        extra_deck_cards = data["extra_deck"]
        extra_deck_count = sum(extra_deck_cards.values())

        main_deck_title = f"## {emoji.effect}{emoji.spell}{emoji.trap} {get_localization_value("yugioh.main_deck", locale=ctx.locale)}[{main_deck_count}/60]"
        monsters_title = f"### {emoji.effect} {get_localization_value("yugioh.monster_card", locale=ctx.locale)} x{monster_cards_count}"
        spell_title = f"### {emoji.spell} {get_localization_value("yugioh.spell_card", locale=ctx.locale)} x{spell_cards_count}"
        trap_title = f"### {emoji.trap} {get_localization_value("yugioh.trap_card", locale=ctx.locale)} x{trap_cards_count}"
        extra_title = f"## {emoji.fusion}{emoji.synchro}{emoji.xyz}{emoji.link} {get_localization_value("yugioh.extra_deck", locale=ctx.locale)}[{extra_deck_count}/15]"
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
            title=f"{get_localization_value("yugioh.deck", locale=ctx.locale)}",
            author=discord.EmbedAuthor(ctx.user.name, icon_url=ctx.user.avatar.url),
            description=content
        )
    
    
    @staticmethod
    def error_message(response: JSONResponse) -> discord.Embed:
        content = response.body.decode()
        try:
            dct = json.loads(content)
        except json.JSONDecodeError:
            pass

        if isinstance(dct, dict):
            description = "\n".join(f"{k}: {v}" for k, v in dct.items())
        else:
            description = content

        return discord.Embed(
            color=discord.Color.red(),
            title=":no_entry: Error",
            description=description
        )
    
    
    @staticmethod
    async def _recognize(image: discord.Attachment, progressor: RecognizeProgresser, other_task: asyncio.Task) -> JSONResponse | dict[str, Any]:
        response = await api_recognize(file=image, progressor=progressor)
        other_task.cancel()
        return response


    @discord.slash_command(
        description=get_localization_value("recognize.description"), 
        description_localizations=get_localizations("recognize.description")
    )
    async def recognize(self, ctx: discord.ApplicationContext, image: discord.Option(discord.Attachment, description=get_localization_value("recognize.desc_image"), description_localizations=get_localizations("recognize.desc_image"))):
        await ctx.defer()
        progressor = RecognizeProgresser(ctx)
        async with asyncio.TaskGroup() as tg:
            t1 = tg.create_task(progressor.trace_progress())
            t2 = tg.create_task(self._recognize(image, progressor, t1))
        response = t2.result()
        del image
        if isinstance(response, JSONResponse):
            return await ctx.edit(embed=self.error_message(response))
        
        data = response["data"]
        await ctx.edit(embed=self._recognize_output(ctx, data))