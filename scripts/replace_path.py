import sys

if len(sys.argv) != 4:
    print("用法：python replace_path.py <file_path> <old_str> <new_str>")
    sys.exit(1)

file_path = sys.argv[1]
old_str = sys.argv[2]
new_str = sys.argv[3]

# 讀取原始內容
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替換字串
updated_content = content.replace(old_str, new_str)

# 寫回原檔
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(updated_content)

print(f"已將 '{old_str}' 替換為 '{new_str}'' in {file_path}")
