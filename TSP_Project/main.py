
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """Hàm chính để khởi động ứng dụng"""
    app = QApplication(sys.argv)
    
    # Thiết lập style
    app.setStyle('Fusion')
    
    # Tạo và hiển thị cửa sổ chính
    window = MainWindow()
    window.show()
    
    # Chạy vòng lặp sự kiện
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
