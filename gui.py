"""
手写数学公式识别系统 - GUI界面
================================

提供图形用户界面进行公式识别。
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import logging
import traceback
import io

# matplotlib 用于 LaTeX 渲染
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 设置基础路径（支持PyInstaller打包）
def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
sys.path.insert(0, BASE_PATH)

from src.preprocessing import ImagePreprocessor
from src.segmentation import SymbolSegmenter
from src.recognition import SymbolRecognizer
from src.structure_analysis import StructureAnalyzer
from src.semantic import SemanticProcessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FormulaRecognizerGUI:
    """公式识别GUI应用"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("手写数学公式识别系统")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # 设置图标（如果存在）
        icon_path = os.path.join(BASE_PATH, 'icon.ico')
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.is_processing = False
        
        # 初始化识别器
        self.preprocessor = None
        self.segmenter = None
        self.recognizer = None
        self.structure_analyzer = None
        self.semantic_processor = None
        
        # 创建界面
        self._create_ui()
        
        # 异步加载模型
        self.root.after(100, self._load_model_async)
    
    def _create_ui(self):
        """创建用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部工具栏
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # 按钮样式
        style = ttk.Style()
        style.configure('Action.TButton', font=('Microsoft YaHei', 10))
        
        # 打开图片按钮
        self.btn_open = ttk.Button(
            toolbar, text="📂 打开图片", 
            command=self._open_image,
            style='Action.TButton',
            width=15
        )
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        # 识别按钮
        self.btn_recognize = ttk.Button(
            toolbar, text="🔍 识别公式", 
            command=self._recognize,
            style='Action.TButton',
            width=15
        )
        self.btn_recognize.pack(side=tk.LEFT, padx=5)
        self.btn_recognize.config(state=tk.DISABLED)
        
        # 清除按钮
        self.btn_clear = ttk.Button(
            toolbar, text="🗑️ 清除", 
            command=self._clear,
            style='Action.TButton',
            width=15
        )
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        # 复制LaTeX按钮
        self.btn_copy = ttk.Button(
            toolbar, text="📋 复制LaTeX", 
            command=self._copy_latex,
            style='Action.TButton',
            width=15
        )
        self.btn_copy.pack(side=tk.LEFT, padx=5)
        self.btn_copy.config(state=tk.DISABLED)
        
        # 状态标签
        self.status_label = ttk.Label(toolbar, text="正在加载模型...", font=('Microsoft YaHei', 9))
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # 中间区域 - 分为左右两部分
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧 - 图片显示区域
        left_frame = ttk.LabelFrame(content_frame, text="输入图像", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 图片画布
        self.canvas = tk.Canvas(left_frame, bg='#f0f0f0', highlightthickness=1, highlightbackground='#ccc')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 拖放提示
        self.canvas.create_text(
            200, 150, 
            text="点击\"打开图片\"按钮\n或粘贴图片到此处",
            font=('Microsoft YaHei', 12),
            fill='#888',
            tags='hint'
        )
        
        # 右侧 - 结果显示区域
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # LaTeX结果
        latex_frame = ttk.LabelFrame(right_frame, text="识别结果 (LaTeX)", padding="5")
        latex_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.latex_text = tk.Text(
            latex_frame, height=3, wrap=tk.WORD,
            font=('Consolas', 12), bg='#fffef0'
        )
        self.latex_text.pack(fill=tk.X)
        self.latex_text.config(state=tk.DISABLED)
        
        # 公式预览（使用 matplotlib 渲染 LaTeX）
        preview_frame = ttk.LabelFrame(right_frame, text="公式预览", padding="5")
        preview_frame.pack(fill=tk.X, pady=5)
        
        # 使用 Canvas 显示渲染后的公式图片
        self.preview_canvas = tk.Canvas(
            preview_frame, 
            height=80, 
            bg='white',
            highlightthickness=1,
            highlightbackground='#ddd'
        )
        self.preview_canvas.pack(fill=tk.X, pady=5)
        self.preview_image = None  # 保持对图片的引用
        
        # 语义分析结果
        semantic_frame = ttk.LabelFrame(right_frame, text="语义分析", padding="5")
        semantic_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.semantic_text = tk.Text(
            semantic_frame, wrap=tk.WORD,
            font=('Microsoft YaHei', 10), bg='#f8f8f8'
        )
        self.semantic_text.pack(fill=tk.BOTH, expand=True)
        self.semantic_text.config(state=tk.DISABLED)
        
        # 底部状态栏
        status_bar = ttk.Frame(main_frame)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
        self.progress = ttk.Progressbar(status_bar, mode='indeterminate', length=200)
        self.progress.pack(side=tk.LEFT)
        
        self.info_label = ttk.Label(status_bar, text="就绪", font=('Microsoft YaHei', 9))
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        # 绑定事件
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        
        # 尝试启用拖放功能
        self._setup_drag_drop()
    
    def _setup_drag_drop(self):
        """设置拖放功能"""
        try:
            # 尝试使用tkinterdnd2（如果安装了）
            try:
                from tkinterdnd2 import DND_FILES, TkinterDnD
                # 如果根窗口支持DnD
                if hasattr(self.root, 'drop_target_register'):
                    self.canvas.drop_target_register(DND_FILES)
                    self.canvas.dnd_bind('<<Drop>>', self._on_drop)
                    logger.info("拖放功能已启用(tkinterdnd2)")
                    return
            except ImportError:
                pass
            
            # Windows原生拖放支持
            # 使用简单的粘贴方式作为替代
            self.root.bind('<Control-v>', self._on_paste)
            
            # 更新提示文字
            self.canvas.delete('hint')
            self.canvas.create_text(
                200, 150, 
                text="点击\"打开图片\"按钮选择图片\n或使用 Ctrl+V 粘贴图片路径",
                font=('Microsoft YaHei', 12),
                fill='#888',
                tags='hint'
            )
            logger.info("使用Ctrl+V粘贴功能")
            
        except Exception as e:
            logger.warning(f"设置拖放功能失败: {e}")
    
    def _on_drop(self, event):
        """处理拖放事件"""
        try:
            # 获取拖放的文件路径
            path = event.data
            # 清理路径（去除大括号等）
            if path.startswith('{') and path.endswith('}'):
                path = path[1:-1]
            path = path.strip()
            
            # 检查是否为图片文件
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self._load_image(path)
            else:
                messagebox.showwarning("警告", "请拖放图片文件（PNG/JPG/BMP）")
        except Exception as e:
            logger.error(f"拖放处理失败: {e}")
    
    def _on_paste(self, event):
        """处理粘贴事件"""
        try:
            # 尝试从剪贴板获取文件路径
            clipboard = self.root.clipboard_get()
            if clipboard and os.path.isfile(clipboard):
                if clipboard.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self._load_image(clipboard)
                    return
            # 尝试获取剪贴板图片
            try:
                from PIL import ImageGrab
                img = ImageGrab.grabclipboard()
                if img is not None:
                    # 保存临时文件
                    temp_path = os.path.join(os.environ.get('TEMP', '.'), 'clipboard_image.png')
                    img.save(temp_path)
                    self._load_image(temp_path)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"粘贴处理: {e}")
    
    def _load_model_async(self):
        """异步加载模型"""
        def load():
            try:
                self.preprocessor = ImagePreprocessor()
                self.segmenter = SymbolSegmenter()
                self.recognizer = SymbolRecognizer()
                self.structure_analyzer = StructureAnalyzer()
                self.semantic_processor = SemanticProcessor()
                
                # 加载模型
                model_path = os.path.join(BASE_PATH, 'models', 'model_2025_12_19_19_05.pkl')
                if os.path.exists(model_path):
                    self.recognizer.load_model(model_path)
                    self.root.after(0, lambda: self._update_status("模型加载完成，就绪"))
                else:
                    # 搜索models目录下的模型
                    models_dir = os.path.join(BASE_PATH, 'models')
                    if os.path.exists(models_dir):
                        for f in os.listdir(models_dir):
                            if f.endswith('.pkl'):
                                model_path = os.path.join(models_dir, f)
                                self.recognizer.load_model(model_path)
                                self.root.after(0, lambda: self._update_status(f"模型加载完成: {f}"))
                                return
                    self.root.after(0, lambda: self._update_status("警告：未找到模型文件"))
            except Exception as e:
                logger.error(f"模型加载失败: {e}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self._update_status(f"模型加载失败: {e}"))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _update_status(self, message):
        """更新状态"""
        self.status_label.config(text=message)
        self.info_label.config(text=message)
    
    def _open_image(self):
        """打开图片"""
        filetypes = [
            ("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("所有文件", "*.*")
        ]
        
        path = filedialog.askopenfilename(
            title="选择公式图片",
            filetypes=filetypes
        )
        
        if path:
            self._load_image(path)
    
    def _load_image(self, path):
        """加载并显示图片"""
        try:
            # 读取图片
            self.current_image = cv2.imread(path)
            if self.current_image is None:
                messagebox.showerror("错误", "无法读取图片文件")
                return
            
            self.current_image_path = path
            
            # 显示图片
            self._display_image()
            
            # 启用识别按钮
            self.btn_recognize.config(state=tk.NORMAL)
            
            # 更新状态
            h, w = self.current_image.shape[:2]
            self._update_status(f"已加载: {os.path.basename(path)} ({w}x{h})")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {e}")
    
    def _display_image(self):
        """在画布上显示图片"""
        if self.current_image is None:
            return
        
        # 清除画布
        self.canvas.delete('all')
        
        # 获取画布大小
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 400, 300
        
        # 转换颜色空间
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # 计算缩放比例
        h, w = img_rgb.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if scale < 1.0:
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为PhotoImage
        pil_img = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        # 居中显示
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
    
    def _on_canvas_resize(self, event):
        """画布大小改变时重新显示图片"""
        if self.current_image is not None:
            self._display_image()
    
    def _recognize(self):
        """识别公式"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先打开一张图片")
            return
        
        if self.recognizer is None or not self.recognizer.is_trained:
            messagebox.showwarning("警告", "模型尚未加载完成，请稍候")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        self.btn_recognize.config(state=tk.DISABLED)
        self.progress.start()
        self._update_status("正在识别...")
        
        def process():
            latex = ""
            semantic = None
            try:
                logger.info("开始识别...")
                
                # 预处理
                logger.info("预处理中...")
                binary = self.preprocessor.process(self.current_image)
                
                # 分割
                logger.info("分割中...")
                symbols = self.segmenter.segment(binary)
                logger.info(f"分割得到 {len(symbols)} 个符号")
                
                if not symbols:
                    self.root.after(0, lambda: self._show_result("(未检测到符号)", None))
                    return
                
                # 识别
                logger.info("识别中...")
                recognized = self.recognizer.recognize_symbols(symbols)
                logger.info(f"识别结果: {recognized}")
                
                # 结构分析
                logger.info("结构分析中...")
                syntax_tree, latex = self.structure_analyzer.analyze(recognized)
                logger.info(f"LaTeX: {latex}")
                
                # 语义分析（可能失败，不影响主结果）
                logger.info("语义分析中...")
                try:
                    semantic = self.semantic_processor.process(latex, syntax_tree)
                except Exception as sem_err:
                    logger.warning(f"语义分析失败: {sem_err}")
                    semantic = {'formula_type': 'expression', 'explanation': '语义分析暂不可用'}
                
                # 更新UI
                logger.info("更新界面...")
                final_latex = latex
                final_semantic = semantic
                self.root.after(0, lambda: self._show_result(final_latex, final_semantic))
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"识别失败: {error_msg}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self._show_error(error_msg))
            finally:
                self.root.after(0, self._finish_processing)
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def _show_result(self, latex, semantic):
        """显示识别结果"""
        # 显示LaTeX
        self.latex_text.config(state=tk.NORMAL)
        self.latex_text.delete('1.0', tk.END)
        self.latex_text.insert('1.0', latex if latex else "(未识别到公式)")
        self.latex_text.config(state=tk.DISABLED)
        
        # 使用 matplotlib 渲染 LaTeX 公式
        self._render_latex_preview(latex if latex else "")
        
        # 显示语义分析
        self.semantic_text.config(state=tk.NORMAL)
        self.semantic_text.delete('1.0', tk.END)
        
        if semantic:
            lines = []
            
            # 处理SemanticResult对象或字典
            if hasattr(semantic, 'formula_type'):
                # SemanticResult 对象
                formula_type = semantic.formula_type
                if hasattr(formula_type, 'value'):
                    formula_type = formula_type.value
                lines.append(f"公式类型: {formula_type}")
                
                if semantic.variables:
                    lines.append(f"变量: {', '.join(semantic.variables)}")
                
                if semantic.constants:
                    lines.append(f"常量: {', '.join(semantic.constants)}")
                
                if semantic.operations:
                    lines.append(f"运算: {', '.join(semantic.operations)}")
                
                if semantic.simplified:
                    lines.append(f"\n化简结果: {semantic.simplified}")
                
                if semantic.interpretation:
                    lines.append(f"\n解释: {semantic.interpretation}")
                
                if semantic.solution:
                    lines.append(f"\n求解结果:")
                    if isinstance(semantic.solution, dict):
                        for k, v in semantic.solution.items():
                            lines.append(f"  {k}: {v}")
                    else:
                        lines.append(f"  {semantic.solution}")
                
                if semantic.errors:
                    lines.append(f"\n错误: {', '.join(semantic.errors)}")
                    
            elif isinstance(semantic, dict):
                # 字典格式
                lines.append(f"公式类型: {semantic.get('formula_type', 'unknown')}")
                
                if semantic.get('variables'):
                    lines.append(f"变量: {', '.join(semantic['variables'])}")
                
                if semantic.get('explanation'):
                    lines.append(f"\n解释: {semantic['explanation']}")
            else:
                lines.append(f"语义信息: {semantic}")
            
            self.semantic_text.insert('1.0', '\n'.join(lines))
        else:
            self.semantic_text.insert('1.0', "无语义信息")
        
        self.semantic_text.config(state=tk.DISABLED)
        
        # 启用复制按钮
        self.btn_copy.config(state=tk.NORMAL)
        
        self._update_status(f"识别完成: {latex}")
    
    def _render_latex_preview(self, latex: str):
        """使用 matplotlib 渲染 LaTeX 公式"""
        # 清除旧内容
        self.preview_canvas.delete('all')
        
        if not latex or latex.strip() == "":
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2 or 200, 40,
                text="(无公式)",
                font=('Microsoft YaHei', 12),
                fill='#888'
            )
            return
        
        try:
            # 预处理 LaTeX 以兼容 matplotlib
            # matplotlib 不支持某些 LaTeX 命令，需要转换
            display_latex = latex
            
            # 替换不支持的命令
            unsupported_replacements = {
                r'\mathds': r'\mathbb',  # mathds 用 mathbb 替代
                r'\mathscr': r'\mathcal',  # mathscr 用 mathcal 替代
                r'\mathfrak': r'\mathrm',  # mathfrak 简化为 mathrm
            }
            for old, new in unsupported_replacements.items():
                display_latex = display_latex.replace(old, new)
            
            # 创建图形
            fig = plt.figure(figsize=(6, 1), dpi=100)
            fig.patch.set_facecolor('white')
            
            # 渲染 LaTeX
            fig.text(0.5, 0.5, f'${display_latex}$', 
                    fontsize=20, 
                    ha='center', va='center',
                    transform=fig.transFigure)
            
            # 转换为图片
            canvas_agg = FigureCanvasAgg(fig)
            canvas_agg.draw()
            
            # 获取图片数据
            buf = canvas_agg.buffer_rgba()
            width, height = fig.canvas.get_width_height()
            img_array = np.asarray(buf).reshape(height, width, 4)
            
            # 转换为 PIL Image
            pil_image = Image.fromarray(img_array[:, :, :3])  # 只取 RGB
            
            # 裁剪白边
            pil_image = self._trim_whitespace(pil_image)
            
            # 调整大小以适应画布
            canvas_height = 70
            if pil_image.height > 0:
                scale = canvas_height / pil_image.height
                new_width = int(pil_image.width * scale)
                pil_image = pil_image.resize((new_width, canvas_height), Image.Resampling.LANCZOS)
            
            # 转换为 Tkinter 可用的格式
            self.preview_image = ImageTk.PhotoImage(pil_image)
            
            # 在画布上显示
            canvas_width = self.preview_canvas.winfo_width() or 400
            self.preview_canvas.create_image(
                canvas_width // 2, 40,
                image=self.preview_image,
                anchor=tk.CENTER
            )
            
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"LaTeX 渲染失败: {e}")
            # 渲染失败时显示原始文本
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2 or 200, 40,
                text=latex,
                font=('Consolas', 14),
                fill='#333'
            )
    
    def _trim_whitespace(self, image: Image.Image) -> Image.Image:
        """裁剪图片周围的白边"""
        # 转换为灰度图
        gray = image.convert('L')
        # 获取边界框
        bbox = gray.getbbox()
        if bbox:
            # 添加一点边距
            padding = 10
            left = max(0, bbox[0] - padding)
            top = max(0, bbox[1] - padding)
            right = min(image.width, bbox[2] + padding)
            bottom = min(image.height, bbox[3] + padding)
            return image.crop((left, top, right, bottom))
        return image

    def _show_error(self, error):
        """显示错误"""
        # 也显示部分结果
        self.latex_text.config(state=tk.NORMAL)
        self.latex_text.delete('1.0', tk.END)
        self.latex_text.insert('1.0', f"(识别出错: {error})")
        self.latex_text.config(state=tk.DISABLED)
        
        self.semantic_text.config(state=tk.NORMAL)
        self.semantic_text.delete('1.0', tk.END)
        self.semantic_text.insert('1.0', f"错误信息:\n{error}")
        self.semantic_text.config(state=tk.DISABLED)
        
        messagebox.showerror("识别失败", f"处理过程中出错:\n{error}")
        self._update_status("识别失败")
    
    def _finish_processing(self):
        """完成处理"""
        self.is_processing = False
        self.btn_recognize.config(state=tk.NORMAL)
        self.progress.stop()
    
    def _clear(self):
        """清除所有内容"""
        self.current_image = None
        self.current_image_path = None
        
        # 清除画布
        self.canvas.delete('all')
        self.canvas.create_text(
            200, 150, 
            text="点击\"打开图片\"按钮\n或将图片拖放到此处",
            font=('Microsoft YaHei', 12),
            fill='#888',
            tags='hint'
        )
        
        # 清除结果
        self.latex_text.config(state=tk.NORMAL)
        self.latex_text.delete('1.0', tk.END)
        self.latex_text.config(state=tk.DISABLED)
        
        # 清除预览画布
        self.preview_canvas.delete('all')
        self.preview_image = None
        
        self.semantic_text.config(state=tk.NORMAL)
        self.semantic_text.delete('1.0', tk.END)
        self.semantic_text.config(state=tk.DISABLED)
        
        # 禁用按钮
        self.btn_recognize.config(state=tk.DISABLED)
        self.btn_copy.config(state=tk.DISABLED)
        
        self._update_status("已清除")
    
    def _copy_latex(self):
        """复制LaTeX到剪贴板"""
        self.latex_text.config(state=tk.NORMAL)
        latex = self.latex_text.get('1.0', tk.END).strip()
        self.latex_text.config(state=tk.DISABLED)
        
        if latex and latex != "(未识别到公式)":
            self.root.clipboard_clear()
            self.root.clipboard_append(latex)
            self._update_status("已复制到剪贴板")
        else:
            messagebox.showinfo("提示", "没有可复制的内容")


def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置DPI感知（Windows）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    app = FormulaRecognizerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
