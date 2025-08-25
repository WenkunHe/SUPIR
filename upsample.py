from PIL import Image
import os
import glob

def upsample_images(input_folder, output_folder, scale_factor=2):
    """
    将输入文件夹中的所有PNG图片放大2倍并保存为JPG到输出文件夹
    
    Args:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
        scale_factor (int): 放大倍数，默认为2
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有PNG文件
    png_files = glob.glob(os.path.join(input_folder, "*.png"))
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    for png_path in png_files:
        try:
            # 打开图片
            with Image.open(png_path) as img:
                # 获取原始尺寸
                original_width, original_height = img.size
                
                # 计算新尺寸
                new_width = original_width * scale_factor
                new_height = original_height * scale_factor
                
                # 使用LANCZOS算法进行高质量放大
                upsampled_img = img.resize(
                    (new_width, new_height), 
                    Image.Resampling.LANCZOS
                )
                
                # 生成输出文件名（将.png替换为.jpg）
                filename = os.path.basename(png_path)
                jpg_filename = os.path.splitext(filename)[0] + ".jpg"
                output_path = os.path.join(output_folder, jpg_filename)
                
                # 保存为JPG，设置高质量参数
                upsampled_img.save(output_path, "JPEG", quality=95)
                
                print(f"处理完成: {filename} -> {jpg_filename}")
                
        except Exception as e:
            print(f"处理文件 {png_path} 时出错: {e}")
    
    print("所有图片处理完成！")

# 使用示例
if __name__ == "__main__":
    input_folder = "output"  # 替换为您的输入文件夹路径
    output_folder = "output2"  # 替换为您的输出文件夹路径
    
    upsample_images(input_folder, output_folder, scale_factor=2)