"""
find-scene.com API 自动化抓取脚本

通过官方 API 搜索电影经典场景，定位台词时间点，下载视频片段。

使用方法:
  1. 去 https://find-scene.com/app 聊天里发 "generate API token" 获取 token
  2. 设置环境变量:  export FIND_SCENE_TOKEN="你的token"
     或者直接修改下面的 API_TOKEN 变量
  3. 运行:
     python scrape_scenes.py              # 仅搜索，保存结果到 scenes.json
     python scrape_scenes.py --download   # 搜索 + 下载视频片段
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

API_BASE = "https://api.find-scene.com"
API_TOKEN = os.environ.get("FIND_SCENE_TOKEN", "")

OUTPUT_JSON = Path(__file__).parent / "scenes.json"
VIDEOS_DIR = Path(__file__).parent / "videos"
DOWNLOAD_ENABLED = "--download" in sys.argv

MOVIES = [
    {
        "name": "Jerry Maguire",
        "search_name": "Jerry Maguire",
        "phrase": "You complete me",
        "year": 1996,
        "scene_desc": None,
    },
    {
        "name": "Notting Hill",
        "search_name": "Notting Hill",
        "phrase": "I'm just a girl, standing in front of a boy, asking him to love her",
        "year": 1999,
        "scene_desc": None,
    },
    {
        "name": "花束般的恋爱",
        "search_name": "We Made a Beautiful Bouquet",
        "phrase": None,
        "year": 2021,
        "scene_desc": "restaurant confession scene",
    },
    {
        "name": "Love Letter",
        "search_name": "Love Letter",
        "phrase": None,
        "year": 1995,
        "scene_desc": "library card scene with the name written on it",
    },
    {
        "name": "重庆森林 Chungking Express",
        "search_name": "Chungking Express",
        "phrase": "0.01",
        "year": 1994,
        "scene_desc": None,
    },
    {
        "name": "花样年华 In the Mood for Love",
        "search_name": "In the Mood for Love",
        "phrase": None,
        "year": 2000,
        "scene_desc": "secret whispered into a hole in the wall at Angkor Wat",
    },
]

CLIP_PADDING_BEFORE = 30  # 台词前后各取多少秒
CLIP_PADDING_AFTER = 30
MAX_CLIPS_PER_MOVIE = 3
POLL_INTERVAL = 5
POLL_TIMEOUT = 300


def time_str_to_seconds(t: str) -> float:
    parts = t.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def seconds_to_time_str(s: float) -> str:
    s = max(0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


async def api_post(client: httpx.AsyncClient, endpoint: str, payload: dict) -> dict:
    payload["_token"] = API_TOKEN
    resp = await client.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


async def api_get(client: httpx.AsyncClient, endpoint: str) -> dict:
    resp = await client.get(f"{API_BASE}{endpoint}", timeout=60)
    resp.raise_for_status()
    return resp.json()


async def get_video_source(client: httpx.AsyncClient, movie: dict) -> str | None:
    """获取电影的 video source hash。"""
    try:
        result = await api_post(client, "/api/get_best_video_source", {
            "query": {
                "movieOrTVShowName": movie["search_name"],
                "year": movie.get("year"),
            }
        })
        sources = result.get("result", {}).get("sources", [])
        if sources:
            return sources[0]
        video_hash = result.get("result", {}).get("hash") or result.get("result", {}).get("videoHash")
        return video_hash
    except Exception as e:
        print(f"    ❌ 获取视频源失败: {e}")
        return None


async def get_text_source(client: httpx.AsyncClient, video_hash: str) -> str | None:
    """获取字幕/文本源 hash。"""
    try:
        result = await api_post(client, "/api/get_high_accuracy_text_source", {
            "videoHash": video_hash,
        })
        return result.get("result", {}).get("textSourceHash") or result.get("result", {}).get("hash")
    except httpx.HTTPStatusError:
        try:
            result = await api_post(client, "/api/get_text_source", {
                "videoHash": video_hash,
            })
            return result.get("result", {}).get("textSourceHash") or result.get("result", {}).get("hash")
        except Exception as e:
            print(f"    ❌ 获取文本源失败: {e}")
            return None
    except Exception as e:
        print(f"    ❌ 获取文本源失败: {e}")
        return None


async def search_by_phrase(
    client: httpx.AsyncClient, text_source: str, phrase: str, max_results: int = 3
) -> list[dict]:
    """通过台词搜索时间点。"""
    try:
        result = await api_post(client, "/api/search_phrase", {
            "phraseSearchParams": {
                "nSkip": 0,
                "maxOccurrences": max_results,
                "phraseStart": phrase,
            },
            "textSource": text_source,
        })
        occurrences = result.get("result", {}).get("occurrences", [])
        return occurrences
    except Exception as e:
        print(f"    ❌ 台词搜索失败: {e}")
        return []


async def search_by_description(
    client: httpx.AsyncClient, movie: dict, description: str
) -> list[dict]:
    """通过场景描述搜索。"""
    try:
        result = await api_post(client, "/api/find_by_scene_description", {
            "description": description,
            "video": {
                "movieOrTVShowName": movie["search_name"],
                "year": movie.get("year"),
            },
        })
        results_list = result.get("result", {}).get("results", [])
        return results_list
    except Exception as e:
        print(f"    ❌ 场景描述搜索失败: {e}")
        return []


async def download_clip(
    client: httpx.AsyncClient,
    video_hash: str,
    text_source: str | None,
    start_time: str,
    end_time: str,
) -> str | None:
    """提交下载任务，返回 operationId。"""
    payload = {
        "startTime": start_time,
        "endTime": end_time,
        "videoHash": video_hash,
    }
    if text_source:
        payload["textSource"] = text_source
    payload["displayParams"] = {"removeWatermark": False, "gif": False}

    try:
        result = await api_post(client, "/api/download_by_time", payload)
        op_id = result.get("operationId") or result.get("result", {}).get("operationId")
        return op_id
    except Exception as e:
        print(f"    ❌ 提交下载任务失败: {e}")
        return None


async def poll_operation(client: httpx.AsyncClient, op_id: str) -> dict | None:
    """轮询异步操作直到完成。"""
    start = time.time()
    while time.time() - start < POLL_TIMEOUT:
        try:
            result = await api_get(client, f"/api/operation/{op_id}")
            status = result.get("status") or result.get("result", {}).get("status")
            if status == "done":
                return result
            if status == "error":
                print(f"    ❌ 操作失败: {result}")
                return None
            print(f"    ⏳ 状态: {status} (已等待 {int(time.time() - start)}s)")
        except Exception as e:
            print(f"    ⚠️ 轮询出错: {e}")
        await asyncio.sleep(POLL_INTERVAL)
    print(f"    ❌ 轮询超时 ({POLL_TIMEOUT}s)")
    return None


async def process_movie(client: httpx.AsyncClient, movie: dict) -> dict:
    """处理单部电影：搜索 → 定位 → 下载。返回结果信息。"""
    info = {"name": movie["name"], "clips": [], "search_results": []}

    # Step 1: 获取视频源
    print(f"  📦 获取视频源: {movie['search_name']}")
    video_hash = await get_video_source(client, movie)
    if not video_hash:
        print(f"  ❌ 未找到视频源，跳过")
        return info
    print(f"  ✅ 视频源: {video_hash[:40]}...")

    # Step 2: 获取文本源
    print(f"  📝 获取字幕源...")
    text_source = await get_text_source(client, video_hash)
    if text_source:
        print(f"  ✅ 字幕源: {text_source[:40]}...")
    else:
        print(f"  ⚠️ 未获取到字幕源，仅可用场景描述搜索")

    # Step 3: 搜索
    occurrences = []
    if movie["phrase"] and text_source:
        print(f"  🔍 搜索台词: \"{movie['phrase']}\"")
        occurrences = await search_by_phrase(client, text_source, movie["phrase"], MAX_CLIPS_PER_MOVIE)
        if occurrences:
            for occ in occurrences:
                t = occ.get("time", "?")
                srt = occ.get("srt", "")[:60]
                print(f"    📌 找到: {t} — {srt}")
                info["search_results"].append({"time": t, "srt": srt})

    if not occurrences and movie.get("scene_desc"):
        print(f"  🔍 搜索场景描述: \"{movie['scene_desc']}\"")
        desc_results = await search_by_description(client, movie, movie["scene_desc"])
        if desc_results:
            for r in desc_results[:MAX_CLIPS_PER_MOVIE]:
                t = r.get("time", "?")
                score = r.get("score", 0)
                print(f"    📌 找到: {t} (score={score})")
                occurrences.append(r)
                info["search_results"].append({"time": t, "score": score})

    if not occurrences:
        print(f"  ⚠️ 未找到匹配的场景")
        return info

    # Step 4: 下载片段（如果启用）
    if DOWNLOAD_ENABLED:
        for i, occ in enumerate(occurrences[:MAX_CLIPS_PER_MOVIE], 1):
            t = occ.get("time", "")
            if not t:
                continue
            center_s = time_str_to_seconds(t)
            start = seconds_to_time_str(center_s - CLIP_PADDING_BEFORE)
            end = seconds_to_time_str(center_s + CLIP_PADDING_AFTER)
            print(f"  ⬇️ [{i}] 下载片段: {start} ~ {end}")

            op_id = await download_clip(client, video_hash, text_source, start, end)
            if op_id:
                print(f"    📋 操作ID: {op_id}")
                result = await poll_operation(client, op_id)
                if result:
                    url = (
                        result.get("url")
                        or result.get("result", {}).get("url")
                        or result.get("result", {}).get("downloadUrl")
                    )
                    if url:
                        info["clips"].append({"time": t, "start": start, "end": end, "url": url})
                        print(f"    ✅ 下载链接: {url[:100]}")

    return info


async def main():
    if not API_TOKEN:
        print("❌ 缺少 API Token！")
        print()
        print("获取方法:")
        print("  1. 打开 https://find-scene.com/app")
        print("  2. 在聊天里发送: generate API token")
        print("  3. 设置环境变量: export FIND_SCENE_TOKEN=\"你的token\"")
        print("  4. 或者直接编辑 scrape_scenes.py 顶部的 API_TOKEN 变量")
        sys.exit(1)

    print("=" * 60)
    print("  find-scene.com API 自动化抓取脚本")
    print("=" * 60)
    print(f"  电影数量: {len(MOVIES)}")
    print(f"  每部最多: {MAX_CLIPS_PER_MOVIE} 个片段")
    print(f"  下载模式: {'开启' if DOWNLOAD_ENABLED else '关闭（加 --download 启用）'}")
    print("=" * 60)

    # 检查配额
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            quota = await api_post(client, "/api/check_quota", {})
            remaining = quota.get("result", {}).get("remaining", "?")
            print(f"\n📊 本月剩余搜索配额: {remaining}")
        except Exception as e:
            print(f"\n⚠️ 无法检查配额: {e}")

        all_results = {}

        for i, movie in enumerate(MOVIES, 1):
            print(f"\n{'='*60}")
            print(f"🎬 [{i}/{len(MOVIES)}] {movie['name']}")
            print(f"{'='*60}")

            try:
                info = await process_movie(client, movie)
                all_results[movie["name"]] = {
                    "search_results": info["search_results"],
                    "clips": info["clips"],
                }
            except Exception as e:
                print(f"  ❌ 处理出错: {e}")
                all_results[movie["name"]] = {"error": str(e)}

            await asyncio.sleep(1)

        # 保存
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到: {OUTPUT_JSON}")

        # 汇总
        print(f"\n{'='*60}")
        print("  📊 抓取汇总")
        print(f"{'='*60}")
        for name, data in all_results.items():
            if "error" in data:
                print(f"  ❌ {name}: 出错 - {data['error'][:50]}")
            else:
                n_results = len(data.get("search_results", []))
                n_clips = len(data.get("clips", []))
                status = f"找到 {n_results} 个时间点"
                if DOWNLOAD_ENABLED:
                    status += f", 下载 {n_clips} 个片段"
                print(f"  {'✅' if n_results else '⚠️'} {name}: {status}")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
