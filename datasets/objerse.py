# from ollama import chat, ChatResponse
# import objaverse
# from tqdm import tqdm

# def is_furniture(name: str, description: str, tags: list[str]) -> bool:
#     prompt = f"""
#         You are an expert on object classification.
#         Determine if the following 3D object is a piece of *furniture* (like a chair, table, bed, cabinet, etc).
#         Respond only with 'Yes' or 'No'.

#         Name: {name}
#         Description: {description}
#         Tags: {', '.join(tags)}
#         """
#     try:
#         response: ChatResponse = chat(model='gemma3:1b', messages=[
#             {'role': 'user', 'content': prompt},
#         ])
#         reply = response.message.content.strip().lower()
#         return reply.startswith("yes")
#     except Exception as e:
#         print(f" Error: {e}")
#         return False

# uids = objaverse.load_uids()

# #  開啟儲存檔案（以追加方式即時寫入）
# with open("furniture_uids.txt", "w") as f_out:
#     with tqdm(total=len(uids), desc="判斷中") as pbar:
#         for uid in uids:
#             try:
#                 anno = objaverse.load_annotations([uid])[uid]
#                 name = anno.get("name", "")
#                 desc = anno.get("description", "")
#                 tags = [t["name"] for t in anno.get("tags", [])]

#                 if is_furniture(name, desc, tags):
#                     f_out.write(uid + "\n")
#                     f_out.flush()  # 立即寫入磁碟
#             except Exception as e:
#                 print(f" Failed to load or process {uid}: {e}")
#             pbar.update(1)

# print("處理完成,結果已儲存在 furniture_uids.txt")


import multiprocessing
import objaverse

# 1. 讀取家具 UID 列表
with open("furniture_uids.txt", "r") as f:
    furniture_uids = [line.strip() for line in f if line.strip()]

# 2. 決定要使用多少進程下載
processes = multiprocessing.cpu_count()

# 3. 只下載這些 UID 對應的 3D 物件
objects = objaverse.load_objects(
    uids=furniture_uids,
    download_processes=processes
)

# objects 現在只包含 furniture_uids.txt 裡面的模型了
print(f"Downloaded {len(objects)} furniture objects.")
