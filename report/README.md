# Samsung Innovation Campus – AI Course
## Capstone Project Final Report LaTeX Template

Bộ template LaTeX chuyên nghiệp, hiện đại và chuẩn hóa dành cho báo cáo **Capstone Project Final Report** thuộc chương trình **Samsung Innovation Campus (SIC) – AI Course**.

Template được thiết kế trung thành với tài liệu gốc của Samsung Electronics Corporate Citizenship Office (`SIC_AI_Capstone Project_Final Report.docx`), kế thừa toàn bộ cấu trúc, typography, bảng đánh giá, trang bìa và bảng màu thương hiệu chuẩn Samsung Blue (`#193DB0`), kết hợp hệ thống font hiện đại **Latin Modern** và hỗ trợ đầy đủ tiếng Việt qua **XeLaTeX**.

---

## 1. Cấu Trúc Thư Mục Project

```text
SIC_AI_Capstone_Report/
│
├── main.tex              # File điều phối chính (cấu trúc tài liệu, TOC, gọi các chương)
├── sicaireport.cls       # Document class tùy chỉnh (chuẩn hóa layout, colors, typography)
├── metadata.tex          # Khai báo thông tin dự án, nhóm, giảng viên, lớp học
├── preamble.tex          # Cấu hình gói mở rộng, biblatex IEEE, hyperref
├── references.bib        # File cơ sở dữ liệu trích dẫn theo chuẩn IEEE
│
├── chapter01.tex         # 1. Introduction (Background, Objectives, Roles, Milestones)
├── chapter02.tex         # 2. Project Execution (Data Acquisition, Training, Workflow, System Design)
├── chapter03.tex         # 3. Results (EDA, Modeling, User Interface, Improvements)
├── chapter04.tex         # 4. Projected Impact (Accomplishments, Future Improvements)
├── chapter05.tex         # 5. Team Member Review and Comment (Team Photo, Member Reviews)
├── chapter06.tex         # 6. Instructor Review and Comment (Bảng điểm & Nhận xét của Giảng viên)
│
├── appendices.tex        # Phụ lục A (Mã nguồn), Phụ lục B (Siêu tham số & Thực nghiệm)
│
├── figures/              # Thư mục chứa hình ảnh và biểu đồ
│   └── samsung_logo.png  # Logo Samsung chính thức trích xuất từ tài liệu gốc
├── tables/               # Thư mục chứa bảng biểu module (nếu tách rời)
├── code/                 # Thư mục chứa source code demo
└── README.md             # Hướng dẫn chi tiết sử dụng và biên dịch
```

---

## 2. Yêu Cầu Cài Đặt Môi Trường

Để biên dịch báo cáo chất lượng cao nhất và hỗ trợ font Unicode tiếng Việt, khuyến nghị cài đặt:

### 2.1. Cài đặt Bản Phân Phối LaTeX
- **Windows**: Cài đặt [MiKTeX](https://miktex.org/download) hoặc [TeX Live](https://www.tug.org/texlive/).
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt-get update
  sudo apt-get install texlive-xetex texlive-latex-extra texlive-science biber
  ```
- **macOS**: Cài đặt [MacTeX](https://www.tug.org/mactex/).

### 2.2. Trình Biên Dịch Yêu Cầu
- **XeLaTeX** (hoặc **LuaLaTeX**): Bắt buộc để sử dụng gói `fontspec` và hệ font Latin Modern.
- **Biber**: Bộ xử lý trích dẫn tài liệu tham khảo cho gói `biblatex` theo chuẩn IEEE.

---

## 3. Quy Trình Biên Dịch (Compilation)

### Cách 1: Sử dụng Dòng Lệnh (Terminal / Command Prompt / PowerShell)
Thực hiện chu trình chuẩn 4 bước để đảm bảo mục lục, danh mục bảng, danh mục hình và trích dẫn IEEE được cập nhật đầy đủ:

```bash
xelatex -interaction=nonstopmode main.tex
biber main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

### Cách 2: Sử dụng VS Code (với LaTeX Workshop)
Cấu hình Recipe trong `.vscode/settings.json`:
```json
{
  "latex-workshop.latex.recipes": [
    {
      "name": "XeLaTeX -> Biber -> XeLaTeX*2",
      "tools": ["xelatex", "biber", "xelatex", "xelatex"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "xelatex",
      "command": "xelatex",
      "args": ["-synctex=1", "-interaction=nonstopmode", "-file-line-error", "%DOC%"]
    },
    {
      "name": "biber",
      "command": "biber",
      "args": ["%DOCFILE%"]
    }
  ]
}
```

---

## 4. Hướng Dẫn Tùy Chỉnh Thông Tin Báo Cáo

Tất cả thông tin cố định và dữ liệu nhóm được tập trung tại file **`metadata.tex`**. Bạn **không cần** sửa trực tiếp trong `main.tex` hay các file chương.

### 4.1. Thay đổi Tiêu đề Dự án (`Project Title`)
Mở file `metadata.tex` và chỉnh sửa lệnh:
```latex
\newcommand{\projectTitle}{Hệ Thống Nhận Diện và Cảnh Báo An Toàn Lao Động Ứng Dụng AI}
```

### 4.2. Thay đổi Tên Nhóm (`Team Name`)
```latex
\newcommand{\teamName}{AI Innovators}
```

### 4.3. Thay đổi Danh sách Thành viên (`Members`)
Cập nhật họ tên 5 thành viên trong nhóm:
```latex
\newcommand{\memberOne}{Nguyễn Văn A (Nhóm trưởng)}
\newcommand{\memberTwo}{Trần Thị B}
\newcommand{\memberThree}{Lê Văn C}
\newcommand{\memberFour}{Phạm Minh D}
\newcommand{\memberFive}{Hoàng Anh E}
```

### 4.4. Thay đổi Ngày nộp Báo cáo (`Date`)
```latex
\newcommand{\projectDate}{15/10/2026}
```

### 4.5. Thay đổi Thông tin Giảng viên và Lớp học
Mặc định hệ thống đã được thiết lập theo yêu cầu:
- **Lớp**: `SIC-AI-N02`
- **Giảng viên**: `TS. Trần Quang Quý`
Nếu cần thay đổi:
```latex
\newcommand{\className}{SIC-AI-N02}
\newcommand{\instructorName}{TS. Trần Quang Quý}
```

---

## 5. Hướng Dẫn Thêm Nội Dung Học Thuật

### 5.1. Thêm Hình Ảnh (`Figures`)
Đặt file ảnh vào thư mục `figures/` (hỗ trợ PNG, JPG, PDF) và chèn vào vị trí tương ứng:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/pipeline_architecture.png}
    \caption{Kiến trúc tổng thể của mô hình Deep Learning}
    \label{fig:pipeline_architecture}
\end{figure}
```
Tham chiếu trong bài viết bằng lệnh: `Figure~\ref{fig:pipeline_architecture}`.

### 5.2. Thêm Bảng Biểu (`Tables`)
Template sử dụng chuẩn `booktabs` và `tabularx` để căn chỉnh tự động:
```latex
\begin{table}[htbp]
    \centering
    \caption{Kết quả đánh giá độ chính xác trên tập kiểm thử}
    \label{tab:evaluation_results}
    \begin{tabularx}{\textwidth}{X c c c}
        \toprule
        \textbf{Mô hình} & \textbf{Độ chính xác (\%)} & \textbf{F1-Score (\%)} & \textbf{Thời gian (ms)} \\
        \midrule
        ResNet-50        & 91.2                      & 90.8                   & 14.5 \\
        Vision Transformer (ViT) & \textbf{95.4}      & \textbf{95.1}          & 22.0 \\
        \bottomrule
    \end{tabularx}
\end{table}
```
Tham chiếu trong bài viết: `Table~\ref{tab:evaluation_results}`.

### 5.3. Thêm Công Thức Toán Học
Sử dụng môi trường `amsmath`:
```latex
\begin{equation}
    \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda \mathcal{L}_{\text{reg}}
    \label{eq:total_loss}
\end{equation}
```
Tham chiếu trong bài viết: `Equation~\eqref{eq:total_loss}`.

### 5.4. Thêm Khối Code (Python / C++)
Template sử dụng gói `listings` với font Latin Modern Mono và số dòng tự động:
```latex
\begin{lstlisting}[language=Python, caption={Hàm tính toán ma trận nhầm lẫn}, label={lst:confusion_matrix}]
from sklearn.metrics import classification_report

def evaluate_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    return report
\end{lstlisting}
```

---

## 6. Hướng Dẫn Trích Dẫn và Quản Lý Tài Liệu Tham Khảo (IEEE Style)

Template tuân thủ nghiêm ngặt chuẩn **IEEE Reference Style** thông qua `biblatex` với `style=ieee`.

### 6.1. Thêm Mục Tài liệu vào `references.bib`
Mở file `references.bib` và thêm mục định dạng BibTeX:
```bibtex
@article{lecun2015deep,
  author    = {Yann LeCun and Yoshua Bengio and Geoffrey Hinton},
  title     = {Deep learning},
  journal   = {Nature},
  volume    = {521},
  number    = {7553},
  pages     = {436--444},
  year      = {2015}
}
```

### 6.2. Trích dẫn trong Nội dung
Sử dụng lệnh `\cite{...}`:
```latex
Theo nghiên cứu của LeCun cùng các cộng sự \cite{lecun2015deep}, học sâu đã mang lại bước đột phá...
```
Các số thứ tự `[1]`, `[2]`, `[3]` sẽ tự động được đánh số và liệt kê ở chương **References** cuối báo cáo.

---

## 7. Các Phần Đặc Biệt Của Samsung Capstone Report

### 7.1. Chapter 5: Team Member Review and Comment
- **Ảnh tập thể nhóm**: Đặt file ảnh tại `figures/team_photo.png` và mở chú thích trong `chapter05.tex`.
- **Nhận xét thành viên**: Bảng ý kiến đóng góp được định dạng với màu nền tiêu đề Samsung Light Blue (`#DEE5FA`) và viền xám Samsung chuẩn.

### 7.2. Chapter 6: Instructor Review and Comment
- Bảng đánh giá của giảng viên giữ đúng 100% cơ cấu trọng số điểm của Samsung Innovation Campus:
  - **IDEA**: 10 điểm
  - **APPLICATION**: 30 điểm
  - **RESULT**: 30 điểm
  - **PROJECT MANAGEMENT**: 10 điểm
  - **PRESENTATION & REPORT**: 20 điểm
  - **TỔNG ĐIỂM (TOTAL)**: 100 điểm
- Khu vực chữ ký giảng viên và ngày chấm điểm được bố trí trang trọng phía dưới bảng.

### 7.3. Bản Quyền & Định Danh Thương Hiệu
- Thông tin bản quyền chính thức được giữ nguyên vẹn ở chân trang bìa:
  *©2025 SAMSUNG. All rights reserved. Samsung Electronics Corporate Citizenship Office.*
- Logo Samsung chính thức được nhúng tự động tại `figures/samsung_logo.png`.
- Tùy chọn tiêu đề mục 2.4: Được chú thích chi tiết sự khác biệt giữa "System Diagram" (trong TOC gốc) và "System Design" (trong nội dung gốc) để người dùng quyết định khi nộp bài.

---
*Template phát triển phục vụ các nhóm sinh viên tham gia khóa học AI – Samsung Innovation Campus.*
