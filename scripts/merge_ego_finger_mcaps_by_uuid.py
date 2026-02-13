"""
MCAP 按 UUID8 自动分组合并工具

功能：
1. 扫描指定目录下的 MCAP 文件
2. 解析文件名格式：DAS-Type_Timestamp_Location_Role_UUID8.mcap
3. 按 UUID8 自动分组
4. 将相同 UUID8 的文件合并到一个 MCAP 文件
5. 输出到当前目录的 merged_mcaps 子目录
"""

from mcap.reader import NonSeekingReader
from mcap.writer import Writer
from typing import List, Dict, Tuple, Optional, NamedTuple
from pathlib import Path
import warnings
import sys
import re
import os
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# 默认配置
# ============================================================================
# 默认输入目录（data/mcap_raw/）
DEFAULT_INPUT_DIR = "./data/mcap_raw"

# 默认输出目录（data/mcap_merged/）
DEFAULT_OUTPUT_DIR = "./data/mcap_merged"


class MCAPMessage(NamedTuple):
    """MCAP 消息结构化数据"""
    timestamp: int
    schema: object
    channel: object
    message: object
    robot_id: int


def remap_topic(original_topic: str, robot_id: int) -> str:
    """重命名 Topic，适配多机器人场景"""
    if robot_id != 0:
        return original_topic.replace("/robot0/", f"/robot{robot_id}/", 1)
    return original_topic


def parse_mcap_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    解析 MCAP 文件名格式：DAS-Type_Timestamp_Location_Role_UUID8.mcap 或 DAS_Type_Timestamp_Location_Role_UUID8.mcap
    并根据 center/left/right 关键词分配 robot_id

    :param filename: 文件名
    :return: 解析结果字典，包含 type, timestamp, location, role, uuid8, robot_id；失败返回 None
    """
    # 匹配格式：DAS[-_]Type_Timestamp_Location_Role_UUID8.mcap（支持 DAS- 和 DAS_）
    pattern = r'^DAS[-_](.+?)_(\d+)_(.+?)_(.+?)_([a-f0-9]{8})\.mcap$'
    match = re.match(pattern, filename)

    if match:
        # 根据文件名中的关键词分配 robot_id
        filename_lower = filename.lower()
        if 'center' in filename_lower:
            robot_id = 0
        elif 'left' in filename_lower:
            robot_id = 1
        elif 'right' in filename_lower:
            robot_id = 2
        else:
            robot_id = 0  # 默认值
        
        return {
            'type': match.group(1),
            'timestamp': match.group(2),
            'location': match.group(3),
            'role': match.group(4),
            'uuid8': match.group(5),
            'robot_id': robot_id  # 新增字段
        }
    return None


def scan_mcap_files(input_dir: str) -> Dict[str, List[Tuple[Path, Dict]]]:
    """
    扫描目录下的 MCAP 文件，按 UUID8 分组

    :param input_dir: 输入目录路径
    :return: 按 UUID8 分组的文件字典 {uuid8: [(file_path, parsed_info), ...]}
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"目录不存在: {input_dir}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"不是目录: {input_dir}")

    # 查找所有 .mcap 文件
    mcap_files = list(input_path.glob("*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"目录中没有找到 .mcap 文件: {input_dir}")

    print(f"找到 {len(mcap_files)} 个 MCAP 文件")

    # 按 UUID8 分组
    grouped_files = defaultdict(list)
    unparseable_files = []

    for mcap_file in mcap_files:
        parsed = parse_mcap_filename(mcap_file.name)
        if parsed:
            uuid8 = parsed['uuid8']
            grouped_files[uuid8].append((mcap_file, parsed))
            # 显示 robot_id 分配信息
            robot_id = parsed['robot_id']
            keyword = "center" if robot_id == 0 else "left" if robot_id == 1 else "right"
            print(f"  ✓ {mcap_file.name} -> UUID8: {uuid8}, RobotID: {robot_id} ({keyword})")
        else:
            unparseable_files.append(mcap_file.name)
            print(f"  ✗ {mcap_file.name} (格式不匹配)")

    if unparseable_files:
        print(f"\n警告：{len(unparseable_files)} 个文件格式不匹配，已跳过")

    print(f"\n按 UUID8 分组结果：")
    for uuid8, files in sorted(grouped_files.items()):
        print(f"  UUID8 {uuid8}: {len(files)} 个文件")
        for file_path, info in files:
            robot_id = info['robot_id']
            keyword = "center" if robot_id == 0 else "left" if robot_id == 1 else "right"
            print(f"    - {file_path.name} → robot{robot_id} ({keyword})")

    return dict(grouped_files)


def read_mcap_messages(
    input_path: str,
    robot_id: int = 0,
    ignore_topics: List[str] = None,
    silent: bool = False
) -> List[MCAPMessage]:
    """
    读取单个 MCAP 文件的消息并过滤

    :param input_path: MCAP 文件路径
    :param robot_id: 机器人 ID（用于 Topic 重命名）
    :param ignore_topics: 忽略的 Topic 列表
    :param silent: 是否静默模式
    :return: 结构化的 MCAP 消息列表
    """
    if ignore_topics is None:
        ignore_topics = []

    messages = []
    input_path_obj = Path(input_path)

    if not silent:
        print(f"  读取: {input_path_obj.name}")

    if not input_path_obj.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    if input_path_obj.suffix != ".mcap":
        warnings.warn(f"文件不是 MCAP 格式: {input_path}")

    with open(input_path, "rb") as f:
        reader = NonSeekingReader(f)
        msg_count = 0
        filtered_count = 0

        for schema, channel, message in reader.iter_messages():
            orig_topic = channel.topic

            # 过滤忽略的 Topic
            if any(orig_topic.endswith(ignore_topic) for ignore_topic in ignore_topics):
                filtered_count += 1
                continue

            # 优先使用 publish_time，无则用 log_time
            sort_timestamp = message.publish_time or message.log_time
            if sort_timestamp is None:
                warnings.warn(
                    f"消息无时间戳，跳过 | Topic: {orig_topic} | ChannelID: {channel.id}"
                )
                filtered_count += 1
                continue

            # 封装结构化消息
            messages.append(
                MCAPMessage(
                    timestamp=sort_timestamp,
                    schema=schema,
                    channel=channel,
                    message=message,
                    robot_id=robot_id
                )
            )
            msg_count += 1

    if not silent:
        print(f"    消息数: {msg_count}")

    return messages


def merge_mcap_files_by_uuid(
    input_dir: str,
    output_dir: str = None,
    ignore_topics: List[str] = None,
    chunk_size: int = 1000
) -> Dict[str, Dict]:
    """
    按 UUID8 自动分组并合并 MCAP 文件

    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径（默认为 input_dir/merged_mcaps）
    :param ignore_topics: 忽略的 Topic 列表
    :param chunk_size: 批量写入的块大小
    :return: 合并结果统计
    """
    if ignore_topics is None:
        ignore_topics = []

    # 设置输出目录
    if output_dir is None:
        output_dir = str(Path(input_dir) / "merged_mcaps")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MCAP 按 UUID8 自动分组合并工具")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 步骤1：扫描并分组
    grouped_files = scan_mcap_files(input_dir)

    if not grouped_files:
        print("没有找到可合并的文件")
        return {}

    # 步骤2：逐个 UUID8 进行合并
    merge_results = {}

    for uuid8, files_info in sorted(grouped_files.items()):
        print(f"\n{'=' * 80}")
        print(f"处理 UUID8: {uuid8} ({len(files_info)} 个文件)")
        print(f"{'=' * 80}")

        # 读取所有文件的消息
        all_messages = []
        for file_path, parsed_info in files_info:
            try:
                # 使用从文件名解析的 robot_id
                robot_id = parsed_info['robot_id']
                keyword = "center" if robot_id == 0 else "left" if robot_id == 1 else "right"
                print(f"  读取: {file_path.name} (关键词识别: robot{robot_id} - {keyword})")

                messages = read_mcap_messages(
                    str(file_path),
                    robot_id=robot_id,  # 使用解析的 robot_id
                    ignore_topics=ignore_topics,
                    silent=True
                )
                all_messages.extend(messages)
            except Exception as e:
                print(f"  错误：读取文件失败 {file_path}: {str(e)}")
                continue

        if not all_messages:
            print(f"  警告：没有读取到任何消息，跳过此 UUID8")
            continue

        # 按时间戳排序
        print(f"\n  排序 {len(all_messages)} 条消息...")
        all_messages.sort(key=lambda x: x.timestamp)

        # 生成输出文件名
        # 格式：merged_UUID8_timestamp.mcap
        first_file_info = files_info[0][1]
        timestamp = first_file_info['timestamp']
        output_filename = f"merged_{uuid8}_{timestamp}.mcap"
        output_file_path = output_path / output_filename

        print(f"  输出文件: {output_filename}")

        # 写入合并文件
        schema_map_global = {}
        channel_map_global = {}
        written_count = 0

        try:
            with open(output_file_path, "wb") as out_f:
                writer = Writer(out_f)
                writer.start()

                pbar = tqdm(all_messages, desc="  写入消息", unit="条", leave=False)
                for msg_struct in pbar:
                    # 1. 重命名 Topic（根据 robot_id）
                    orig_topic = msg_struct.channel.topic
                    new_topic = remap_topic(orig_topic, msg_struct.robot_id)

                    # 2. 注册 Schema
                    schema = msg_struct.schema
                    schema_key = (schema.name, schema.encoding, schema.data)
                    if schema_key not in schema_map_global:
                        schema_id = writer.register_schema(
                            name=schema.name,
                            encoding=schema.encoding,
                            data=schema.data
                        )
                        schema_map_global[schema_key] = schema_id
                    schema_id = schema_map_global[schema_key]

                    # 3. 注册 Channel（使用重命名后的 Topic）
                    channel = msg_struct.channel
                    channel_key = (new_topic, channel.message_encoding, schema_key)
                    if channel_key not in channel_map_global:
                        channel_id = writer.register_channel(
                            schema_id=schema_id,
                            topic=new_topic,
                            message_encoding=channel.message_encoding,
                            metadata=channel.metadata
                        )
                        channel_map_global[channel_key] = channel_id
                    channel_id = channel_map_global[channel_key]

                    # 4. 写入消息
                    message = msg_struct.message
                    writer.add_message(
                        channel_id=channel_id,
                        log_time=message.log_time,
                        data=message.data,
                        publish_time=message.publish_time
                    )
                    written_count += 1

                    if written_count % chunk_size == 0:
                        pbar.set_postfix({"已写入": written_count})

                writer.finish()
            pbar.close()

            # 计算时长
            duration = 0.0
            if all_messages:
                duration = (all_messages[-1].timestamp - all_messages[0].timestamp) / 1e9

            print(f"  ✓ 合并完成")
            print(f"    消息数: {written_count}")
            print(f"    时长: {duration:.2f}s")
            print(f"    文件大小: {output_file_path.stat().st_size / (1024 * 1024):.2f} MB")

            merge_results[uuid8] = {
                "output_file": str(output_file_path),
                "message_count": written_count,
                "input_file_count": len(files_info),
                "duration": duration,
                "file_size_mb": output_file_path.stat().st_size / (1024 * 1024)
            }

        except Exception as e:
            print(f"  ✗ 合并失败: {str(e)}")
            if output_file_path.exists():
                output_file_path.unlink()
            continue

    # 打印最终统计
    print(f"\n{'=' * 80}")
    print("合并完成统计")
    print(f"{'=' * 80}")
    print(f"成功合并的 UUID8 数: {len(merge_results)}")
    for uuid8, result in sorted(merge_results.items()):
        print(f"\n  UUID8: {uuid8}")
        print(f"    输入文件数: {result['input_file_count']}")
        print(f"    输出文件: {Path(result['output_file']).name}")
        print(f"    消息数: {result['message_count']}")
        print(f"    时长: {result['duration']:.2f}s")
        print(f"    文件大小: {result['file_size_mb']:.2f} MB")

    print(f"\n输出目录: {output_dir}")

    return merge_results


def main():
    """主函数"""
    # 解析命令行参数
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    if len(sys.argv) >= 2:
        input_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()

    try:
        merge_mcap_files_by_uuid(
            input_dir=input_dir,
            output_dir=output_dir,
            ignore_topics=[],
            chunk_size=1000
        )
    except Exception as e:
        print(f"\n错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
