import re
import threading

import geojson
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QWidget, QMainWindow, QMessageBox, QApplication, QFileDialog, QLineEdit, QLabel, QSlider, \
    QCheckBox, QButtonGroup, QRadioButton, QPushButton, QComboBox, QSpinBox, QVBoxLayout
from AlgorithmList import AlgorithmList
from GridWidget import GridWidget
from MapPygame import PygameWidget
from result import load_demo, Category_Demo, Category_Compare, load_look


class Ui_MainWindow(object):
    windows = []  # 存储所有创建的窗口实例

#动态障碍物窗口，用户选择生成位置，移动方向，
    def DynamicObstaclesUI(self, MainWindow, grid_widget):
        self.loginWindow_new = None
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(500, 300)  # 窗口大小锁定
        # 障碍物形状
        label_shape = QLabel("障碍物形状:", MainWindow)
        label_shape.setGeometry(60, 30, 100, 30)
        shape_group = QButtonGroup(MainWindow)
        radio_circle = QRadioButton("圆形", MainWindow)
        radio_circle.setGeometry(160, 30, 60, 30)
        radio_square = QRadioButton("正方形", MainWindow)
        radio_square.setGeometry(230, 30, 70, 30)

        shape_group.addButton(radio_circle)
        shape_group.addButton(radio_square)


        # 障碍物运动方向
        label_direction = QLabel("运动方向:", MainWindow)
        label_direction.setGeometry(60, 80, 100, 30)
        direction_group = QButtonGroup(MainWindow)
        radio_up = QRadioButton("向上", MainWindow)
        radio_up.setGeometry(160, 80, 60, 30)
        radio_down = QRadioButton("向下", MainWindow)
        radio_down.setGeometry(230, 80, 60, 30)
        radio_left = QRadioButton("向左", MainWindow)
        radio_left.setGeometry(300, 80, 60, 30)
        radio_right = QRadioButton("向右", MainWindow)
        radio_right.setGeometry(370, 80, 60, 30)
        direction_group.addButton(radio_up)
        direction_group.addButton(radio_down)
        direction_group.addButton(radio_left)
        direction_group.addButton(radio_right)

        # 动态障碍物速度
        label_speed = QLabel("运动速度:", MainWindow)
        label_speed.setGeometry(60, 130, 100, 30)
        slider_speed = QSlider(MainWindow)
        slider_speed.setGeometry(160, 130, 200, 30)
        slider_speed.setMinimum(1)
        slider_speed.setMaximum(10)
        slider_speed.setOrientation(Qt.Horizontal)
        label_speed_value = QLabel("1", MainWindow)
        label_speed_value.setGeometry(370, 130, 40, 30)
        slider_speed.valueChanged.connect(lambda value: label_speed_value.setText(str(value)))

        # 按钮：选择动态障碍物位置
        button_select_position = QPushButton("选择动态障碍物位置", MainWindow)
        button_select_position.setGeometry(180, 200, 150, 40)
        button_select_position.clicked.connect(
            lambda: self.select_obstacle_position(MainWindow, grid_widget, shape_group, direction_group, slider_speed))

    def select_obstacle_position(self, settings_window, grid_widget, shape_group, direction_group, slider_speed):
        # 隐藏设置窗口
        settings_window.hide()

        # 提示用户点击位置
        label_notice = QLabel("点击网格选择动态障碍物位置", grid_widget)
        label_notice.setGeometry(10, 10, 200, 30)
        label_notice.setStyleSheet("color: blue")
        label_notice.show()

        # 鼠标点击事件绑定
        def on_grid_click(event):
            # 获取点击位置
            x = event.pos().x()
            y = event.pos().y()
            label_notice.hide()
            grid_widget.mousePressEvent = None  # 取消绑定事件

            # 获取形状、方向、速度
            shape_button = shape_group.checkedButton()
            direction_button = direction_group.checkedButton()
            speed = slider_speed.value()

            if not shape_button or not direction_button:
                QMessageBox.warning(grid_widget, "错误", "请设置障碍物的形状和运动方向")
                return

            shape = shape_button.text()
            direction = direction_button.text()

            # 创建动态障碍物
            grid_widget.create_dynamic_obstacle( x, y, shape, direction, speed)

        grid_widget.mousePressEvent = on_grid_click


    # 参数障碍物窗口 根据用户选择生成相同大小障碍物
    def modify_obstacles(self, MainWindow, grid_widget):
        self.loginWindow_new = None
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(400, 300) # 窗口大小锁定
        # 创建障碍物数量标签
        self.label_sum = QLabel("障碍物数量:", MainWindow)
        self.label_sum.setGeometry(60, 50, 80, 30)  # 设置标签位置和大小
        # 创建第一个输入框
        self.lineEdit1 = QLineEdit(MainWindow)
        self.lineEdit1.setGeometry(140, 50, 200, 30)  # 设置输入框位置和大小
        self.lineEdit1.setPlaceholderText("请输入障碍物的数量")  # 设置提示文字

        # 创建障碍物大小标签
        label_size = QLabel("障碍物大小:", MainWindow)
        label_size.setGeometry(60, 100, 80, 30)  # 设置标签位置和大小
        # # 创建第二个输入框
        # self.lineEdit2 = QLineEdit(MainWindow)
        # self.lineEdit2.setGeometry(140, 100, 200, 30)  # 设置输入框位置和大小
        # self.lineEdit2.setPlaceholderText("迭代次数")  # 设置提示文字
        # 创建障碍物大小滑块
        slider = QSlider(MainWindow)
        slider.setGeometry(140, 100, 200, 30)  # 设置滑块位置和大小
        slider.setMinimum(5)  # 设置最小值
        slider.setMaximum(200)  # 设置最大值
        slider.setOrientation(Qt.Horizontal)  # 设置水平方向

        # 创建标签用于显示滑块的值
        label_slider_value = QLabel("5", MainWindow)
        label_slider_value.setGeometry(350, 100, 40, 30)  # 设置标签位置和大小

        # 连接滑块数值改变的信号和槽
        slider.valueChanged.connect(lambda value: label_slider_value.setText(str(value)))
        # 创建障碍物大小标签
        label_type = QLabel("障碍物类型:", MainWindow)
        label_type.setGeometry(60, 150, 80, 30)  # 设置标签位置和大小

        # 创建多选框
        checkbox1 = QCheckBox("圆形", MainWindow)
        checkbox1.setGeometry(140, 150, 150, 30)  # 设置多选框位置和大小

        checkbox2 = QCheckBox("正方形", MainWindow)
        checkbox2.setGeometry(210, 150, 150, 30)  # 设置多选框位置和大小

        checkbox3 = QCheckBox("椭圆", MainWindow)
        checkbox3.setGeometry(280, 150, 150, 30)  # 设置多选框位置和大小

        checkbox4 = QCheckBox("菱形", MainWindow)
        checkbox4.setGeometry(140, 180, 150, 30)  # 设置多选框位置和大小

        checkbox5 = QCheckBox("矩形", MainWindow)
        checkbox5.setGeometry(210, 180, 150, 30)  # 设置多选框位置和大小

        # 不规则障碍物
        checkbox6 = QCheckBox("不规则", MainWindow)
        checkbox6.setGeometry(280, 180, 150, 30)

        # 创建障碍物是否重叠
        label_type = QLabel("障碍物重叠:", MainWindow)
        label_type.setGeometry(60, 210, 80, 30)  # 设置标签位置和大小

        # 创建单选按钮组
        radio_button_group = QButtonGroup(MainWindow)
        radio_button_group.setExclusive(True)  # 设置为互斥，只能选择一个单选按钮
        # 创建是单选按钮
        radio_button_ok = QRadioButton("重叠", MainWindow)
        radio_button_ok.setGeometry(140, 210, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_ok)  # 将单选按钮添加到单选按钮组
        # 创建是单选按钮
        radio_button_no = QRadioButton("分离", MainWindow)
        radio_button_no.setGeometry(210, 210, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_no)  # 将单选按钮添加到单选按钮组

        # 创建生成障碍物按钮
        self.button_generate_obstacle = QPushButton("生成障碍物", MainWindow)
        self.button_generate_obstacle.setGeometry(130, 250, 120, 30)  # 设置按钮位置和大小

        label_notice = QLabel("", MainWindow)
        label_notice.setGeometry(140, 15, 150, 30)  # 设置标签位置和大小

        def on_button_click():
            obstacle_quantity = self.lineEdit1.text()
            if not obstacle_quantity:
                label_notice.setText("请输入障碍物数量！")
                return
            obstacle_size = slider.value()
            obstacle_types = []
            if not checkbox1.isChecked() and not checkbox2.isChecked() and not checkbox3.isChecked() and not checkbox4.isChecked() and not checkbox5.isChecked()and not checkbox6.isChecked():
                label_notice.setText("请选择至少一种障碍物类型！")
                return
            if checkbox1.isChecked():
                obstacle_types.append(0)
            if checkbox2.isChecked():
                obstacle_types.append(1)
            if checkbox3.isChecked():
                obstacle_types.append(2)
            if checkbox4.isChecked():
                obstacle_types.append(3)
            if checkbox5.isChecked():
                obstacle_types.append(4)
            if checkbox6.isChecked():
                obstacle_types.append(5)
            if not radio_button_ok.isChecked() and not radio_button_no.isChecked():
                label_notice.setText("请选择障碍物是否重叠！")
                return
            if radio_button_ok.isChecked():
                obstacle_overlap = "T"
            elif radio_button_no.isChecked():
                obstacle_overlap = "F"
            grid_widget.graph_setting(obstacle_quantity, obstacle_size, obstacle_types, obstacle_overlap)
            label_notice.setText("障碍物生成成功！")

        # 连接按钮的点击信号
        self.button_generate_obstacle.clicked.connect(on_button_click)

    # TODO 开始规划选择算法窗口
    def start_path(self, MainWindow, grid_widget):
        self.loginWindow_new = None
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(400, 300)
        # 创建障选择算法标签
        self.label_sum = QLabel("规划算法:", MainWindow)
        self.label_sum.setGeometry(60, 50, 80, 30)  # 设置标签位置和大小

        # 创建多选框
        checkbox1 = QCheckBox("A*算法", MainWindow)
        checkbox1.setGeometry(120, 50, 150, 30)  # 设置多选框位置和大小

        checkbox2 = QCheckBox("RRT算法", MainWindow)
        checkbox2.setGeometry(180, 50, 150, 30)  # 设置多选框位置和大小

        checkbox2 = QCheckBox("APF算法", MainWindow)
        checkbox2.setGeometry(250, 50, 150, 30)  # 设置多选框位置和大小
        # 创建是否分析算法结果标签
        self.label_analyse = QLabel("是否分析:", MainWindow)
        self.label_analyse.setGeometry(60, 100, 80, 30)  # 设置标签位置和大小
        # 创建单选按钮组
        radio_button_group = QButtonGroup(MainWindow)
        radio_button_group.setExclusive(True)  # 设置为互斥，只能选择一个单选按钮
        # 创建是单选按钮
        radio_button_ok = QRadioButton("是", MainWindow)
        radio_button_ok.setGeometry(120, 100, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_ok)  # 将单选按钮添加到单选按钮组
        # 创建是单选按钮
        radio_button_no = QRadioButton("否", MainWindow)
        radio_button_no.setGeometry(180, 100, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_no)  # 将单选按钮添加到单选按钮组

        # 单次路径结果比较分析
        button_single = QPushButton("单次路径结果比较分析", MainWindow)
        button_single.setGeometry(10, 10, 50, 30)
        def on_single_click():
            # 创建文件对话框
            dialog = QFileDialog()
            # 设置对话框标题
            dialog.setWindowTitle('打开结果文件')
            # 设置文件过滤器
            dialog.setNameFilter('Text Files (*.txt)')
            # 设置默认文件名，包含文件类型后缀
            # dialog.setDefaultSuffix('txt')
            files, _ = dialog.getOpenFileNames()
            for index, f in enumerate(files):  # 循环选中的所有文件
                self.open_result_single(index, f)
        button_single.clicked.connect(on_single_click)

        # 多路径比较分析
        button_category = QPushButton("多次路径比较分析", MainWindow)
        button_category.setGeometry(10, 40, 50, 30)
        def on_category_click():
            dialog = QFileDialog()
            dialog.setWindowTitle("选择要进行多次结果分析的文件夹")
            dialog.setFileMode(QFileDialog.DirectoryOnly)
            if dialog.exec_():
                category = Category_Demo()  # 创建多结果类
                category.read_file(dialog.selectedFiles()[0])  # 获取选择的文件夹的路径
                self.open_result_category(category)
        button_category.clicked.connect(on_category_click)
        # 创建开始规划按钮
        self.button_start = QPushButton("开始规划", MainWindow)
        self.button_start.setGeometry(130, 250, 120, 30)  # 设置按钮位置和大小

        # 不同算法性能对比(多个多次路径比较分析)
        button_category_compare = QPushButton("不同算法性能比较", MainWindow)
        button_category_compare.setGeometry(10, 70, 50, 30)
        def on_category_compare_click():
            dialog = QFileDialog()
            dialog.setWindowTitle("选择要进行性能对比的文件")
            files, _ = dialog.getOpenFileNames()
            if files:
                compare = Category_Compare()
                compare.read_category(files)
                self.open_result_category_compare(compare)
        button_category_compare.clicked.connect(on_category_compare_click)


    # 随机障碍物提示输入障碍物数量窗口
    def random_ob(self, MainWindow, grid_widget):
        self.loginWindow_new = None
        MainWindow.setObjectName("随机障碍物")
        MainWindow.setFixedSize(300, 120)
        # 创建障碍物数量标签
        self.label_sum = QLabel("障碍物数量:", MainWindow)
        self.label_sum.setGeometry(10, 20, 80, 30)  # 设置标签位置和大小
        # 创建第一个输入框
        self.lineEdit1 = QLineEdit(MainWindow)
        self.lineEdit1.setGeometry(80, 20, 200, 30)  # 设置输入框位置和大小
        self.lineEdit1.setPlaceholderText("请输入障碍物的数量")  # 设置提示文字
        # 创建生成障碍物按钮
        self.button_generate_obstacle = QPushButton("生成障碍物", MainWindow)
        self.button_generate_obstacle.setGeometry(90, 60, 120, 30)  # 设置按钮位置和大小
        label_notice = QLabel("", MainWindow)
        label_notice.setGeometry(90, 90, 150, 30)  # 设置标签位置和大小

        def on_button_click():
            obstacle_quantity = self.lineEdit1.text()
            if not obstacle_quantity:
                label_notice.setText("请输入障碍物数量！")
                return
            grid_widget.random_graph_new(int(obstacle_quantity))
            label_notice.setText("障碍物生成成功！")
        # 连接按钮的点击信号
        self.button_generate_obstacle.clicked.connect(on_button_click)

    # 路径规划选择算法
    def select_arithmetic(self, MainWindow, grid_widget):
        self.loginWindow_new = None
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 250)
        # 创建选择算法标签
        self.label_analyse = QLabel("选择算法:", MainWindow)
        self.label_analyse.setGeometry(30, 30, 80, 30)  # 设置标签位置和大小
        # 创建单选按钮组
        radio_button_group = QButtonGroup(MainWindow)
        radio_button_group.setExclusive(True)  # 设置为互斥，只能选择一个单选按钮
        # 创建A*算法单选按钮
        radio_button_astar = QRadioButton("AStar", MainWindow)
        radio_button_astar.setGeometry(90, 30, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_astar)  # 将单选按钮添加到单选按钮组
        # 创建RRT算法单选按钮
        radio_button_rrt = QRadioButton("RRT", MainWindow)
        radio_button_rrt.setGeometry(155, 30, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_rrt)  # 将单选按钮添加到单选按钮组
        # 创建APF算法单选按钮
        radio_button_apf = QRadioButton("APF", MainWindow)
        radio_button_apf.setGeometry(210, 30, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_apf)  # 将单选按钮添加到单选按钮组
        # 创建RRT-APF算法单选按钮
        radio_button_RRTapf = QRadioButton("RRT-APF", MainWindow)
        radio_button_RRTapf.setGeometry(270, 30, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_RRTapf)  # 将单选按钮添加到单选按钮组
        #PRM
        radio_button_PRM = QRadioButton("PRM", MainWindow)
        radio_button_PRM.setGeometry(370, 30, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_PRM)
        #RRTstar
        radio_button_RRTStar = QRadioButton("RRTStar", MainWindow)
        radio_button_RRTStar.setGeometry(430, 30, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_RRTStar)
        # BiRRT
        radio_button_BiRRT = QRadioButton("BiRRT", MainWindow)
        radio_button_BiRRT.setGeometry(500, 30, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_BiRRT)
        #dynapfrrt
        radio_button_RRTapf_dyn = QRadioButton("RRTAPFdyn", MainWindow)
        radio_button_RRTapf_dyn.setGeometry(560, 30, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_RRTapf_dyn)
        # dbvsapfrrt
        radio_button_dbvspRRT = QRadioButton("dbvspRRT", MainWindow)
        radio_button_dbvspRRT.setGeometry(620, 30, 80, 30)  # 设置单选按钮位置和大小
        radio_button_group.addButton(radio_button_dbvspRRT)
        # Cost-Aware RRT
        radio_button_costRRT = QRadioButton("CostRRT", MainWindow)
        radio_button_costRRT.setGeometry(700, 30, 80, 30)  # 设置位置和大小（x根据你的界面布局可调整）
        radio_button_group.addButton(radio_button_costRRT)
        # 保存结果部分
        label_result = QLabel("是否保存结果:", MainWindow)
        label_result.setGeometry(30, 60, 80, 30)
        radio_button_result = QButtonGroup(MainWindow)
        radio_button_result.setExclusive(True)
        radio_button_result_true = QRadioButton("是", MainWindow)
        radio_button_result_true.setGeometry(110, 60, 80, 30)
        radio_button_result.addButton(radio_button_result_true)
        radio_button_result_false = QRadioButton("否", MainWindow)
        radio_button_result_false.setGeometry(170, 60, 80, 30)
        radio_button_result_false.setChecked(True)  # 默认是不保存结果
        radio_button_result.addButton(radio_button_result_false)
        # 运行次数部分
        label_run = QLabel("是否运行多次:", MainWindow)
        label_run.setGeometry(30, 90, 80, 30)
        radio_button_run = QButtonGroup(MainWindow)
        radio_button_run.setExclusive(True)
        radio_button_run_true = QRadioButton("是", MainWindow)
        radio_button_run_true.setGeometry(110, 90, 80, 30)
        radio_button_run.addButton(radio_button_run_true)
        radio_button_run_false = QRadioButton("否", MainWindow)
        radio_button_run_false.setGeometry(170, 90, 80, 30)
        radio_button_run_false.setChecked(True)  # 默认是只运行一次
        radio_button_run.addButton(radio_button_run_false)
        # 设置结果文件名称的部分
        label_path = QLabel("",MainWindow)
        label_path.setGeometry(30, 120, 200, 30)
        line_edit_result = QLineEdit("", MainWindow)
        line_edit_result.setPlaceholderText("请输入文件名")
        line_edit_result.setGeometry(90, 150, 120, 30)
        line_edit_result.hide()  # 默认是不保存结果,所以先隐藏
        # 选择运行次数部分
        spin_box = QSpinBox(MainWindow)  # 数字选择框
        spin_box.setGeometry(30, 180, 50, 30)
        spin_box.setMinimum(0)  # 设置最小值
        spin_box.setMaximum(100)  # 设置最大值
        spin_box.setValue(1)  # 设置默认值
        spin_box.hide()  # 默认是运行一次,所以先隐藏
        def on_result_click(button):
            if button == radio_button_result_true:
                select_folder = QFileDialog.getExistingDirectory(MainWindow, "选择文件夹", "")
                label_path.setText(select_folder)
                line_edit_result.show()
            else:
                line_edit_result.hide()
        radio_button_result.buttonClicked.connect(on_result_click)
        def on_run_click(button):
            if button == radio_button_run_true:
                spin_box.show()
            else:
                spin_box.hide()
        radio_button_run.buttonClicked.connect(on_run_click)
        # 创建生成障碍物按钮
        self.button_generate_obstacle = QPushButton("开始规划", MainWindow)
        self.button_generate_obstacle.setGeometry(90, 180, 120, 30)  # 设置按钮位置和大小

        label_notice = QLabel("", MainWindow)
        label_notice.setGeometry(80, 210, 150, 30)  # 设置标签位置和大小

        def on_button_click():
            if not radio_button_astar.isChecked() and not radio_button_rrt.isChecked() and not radio_button_apf.isChecked() and not radio_button_RRTapf and not radio_button_PRM and not radio_button_BiRRT and not radio_button_RRTStar and not radio_button_RRTapf_dyn and not radio_button_dbvspRRT and not radio_button_costRRT :
                label_notice.setText("请选择规划路径所用算法！")
                return
            # 检测路径是否合法
            if radio_button_result_true.isChecked() and label_path.text() == "":
                label_notice.setText("请选择文件!")
                radio_button_result_false.setChecked(True)
                return
            # 获取循环次数
            if radio_button_run_true.isChecked():
                spin_box_value = spin_box.value()
            else:
                spin_box_value = 1
            for i in range(spin_box_value):
                track = []  # 算法执行所得的路径
                time = None  # 算法花费的时间
                if radio_button_astar.isChecked():
                    obstacle_overlap = "Astar"
                    track, time = grid_widget.startAstar()
                elif radio_button_rrt.isChecked():
                    obstacle_overlap = "RRT"
                    track, time = grid_widget.startRtt()
                elif radio_button_apf.isChecked():
                    obstacle_overlap = "APF"
                    track, time = grid_widget.startApf()
                elif radio_button_RRTapf.isChecked():
                    obstacle_overlap = "RRT-APF"
                    track, time = grid_widget.startApfRrt()
                elif radio_button_PRM.isChecked():
                    obstacle_overlap = "PRM"
                    track, time = grid_widget.startPRm()
                elif radio_button_RRTStar.isChecked():
                    obstacle_overlap = "RRTStar"
                    track, time = grid_widget.startRRTStar()
                elif radio_button_BiRRT.isChecked():
                    obstacle_overlap = "BiRRT"
                    track, time = grid_widget.startBiRRT()
                elif radio_button_RRTapf_dyn.isChecked():
                    obstacle_overlap = "RRT-APF-Dyn"
                    track, time = grid_widget.startApfRrt_dyn()
                elif radio_button_costRRT .isChecked():
                    obstacle_overlap = "costRRT"
                    track, time = grid_widget.startcostRrt()
                elif radio_button_dbvspRRT.isChecked():
                    obstacle_overlap = "dbvspRRT"
                    track, time = grid_widget.startDbvsPRrt()
                label_notice.setText("正在规划路径！")
                # 保存结果部分
                if radio_button_result_true.isChecked():
                    filepath = label_path.text()+"/"+line_edit_result.text()+str(i)+".txt"
                    grid_widget.save_result(time1=time, track=track, file_path=filepath)

        # 连接按钮的点击信号
        self.button_generate_obstacle.clicked.connect(on_button_click)

    # 地图分辨率调整
    def map(self, MainWindow, grid_widget):
        self.loginWindow_new = None
        MainWindow.setObjectName("地图参数")
        MainWindow.setFixedSize(300, 120)
        # 创建障碍物数量标签
        self.label_sum = QLabel("分辨率:", MainWindow)
        self.label_sum.setGeometry(10, 20, 80, 30)  # 设置标签位置和大小
        # 创建第一个输入框
        self.lineEdit1 = QLineEdit(MainWindow)
        self.lineEdit1.setGeometry(80, 20, 200, 30)  # 设置输入框位置和大小
        self.lineEdit1.setPlaceholderText("请输入分辨率大小")  # 设置提示文字
        # 创建生成障碍物按钮
        self.button_modify = QPushButton("调整", MainWindow)
        self.button_modify.setGeometry(90, 60, 60, 30)  # 设置按钮位置和大小
        self.button_default = QPushButton("默认", MainWindow)
        self.button_default.setGeometry(160, 60, 60, 30)  # 设置按钮位置和大小
        label_notice = QLabel("", MainWindow)
        label_notice.setGeometry(90, 90, 150, 30)  # 设置标签位置和大小

        def on_button_click():
            obstacle_quantity = self.lineEdit1.text()
            if not obstacle_quantity:
                label_notice.setText("请输入地图分辨率！")
                return
            grid_widget.modifyMap(int(obstacle_quantity))
            label_notice.setText("分辨率修改成功！")
        def on_button_default():
            print("默认地图")
        # 连接按钮的点击信号
        self.button_modify.clicked.connect(on_button_click)

    # 起始点输入窗口
    def input_startAndEnd(self, MainWindow, grid_widget):
        self.loginWindow_new = None
        MainWindow.setObjectName("地图起始点")
        MainWindow.setFixedSize(300, 160)
        # 创建起点标签
        self.label_start = QLabel("起点:", MainWindow)
        self.label_start.setGeometry(10, 20, 80, 30)  # 设置标签位置和大小
        # 创建第一个输入框
        self.lineEdit1 = QLineEdit(MainWindow)
        self.lineEdit1.setGeometry(80, 20, 200, 30)  # 设置输入框位置和大小
        self.lineEdit1.setPlaceholderText("请输入起点（x，y）")  # 设置提示文字
        # 创建起点标签
        self.label_end = QLabel("终点:", MainWindow)
        self.label_end.setGeometry(10, 60, 80, 30)  # 设置标签位置和大小
        # 创建第一个输入框
        self.lineEdit2 = QLineEdit(MainWindow)
        self.lineEdit2.setGeometry(80, 60, 200, 30)  # 设置输入框位置和大小
        self.lineEdit2.setPlaceholderText("请输入起点（x，y）")  # 设置提示文字

        # 创建生成起始点按钮
        self.button_modify = QPushButton("生成", MainWindow)
        self.button_modify.setGeometry(120, 100, 60, 30)  # 设置按钮位置和大小
        label_notice = QLabel("", MainWindow)
        label_notice.setGeometry(100, 120, 150, 30)  # 设置标签位置和大小

        def on_button_click():
            start = self.lineEdit1.text()
            end = self.lineEdit2.text()

            # TODO 起始点坐标获取以及调用方法未完成
            if not start and not end:
                label_notice.setText("请输入起始点坐标！")
                return

            print(start)
            pattern = r"\((\d+),(\d+)\)"  # 匹配坐标的正则表达式模式
            match = re.match(pattern, start)
            print(match)
            if match:
                keyx = int(match.group(1))  # 提取横坐标
                keyy = int(match.group(2))  # 提取纵坐标
                print("起点横坐标:", keyx)
                print("起点纵坐标:", keyy)
                grid_widget.painting_ori(keyx, keyy)  # 目前执行到这里程序就结束了，应该是调用有问题
            else:
                print("坐标格式不正确")
            match_2 = re.match(pattern, end)
            if match_2:
                # global keyx_2, keyy_2
                keyx_2 = int(match_2.group(1))  # 提取横坐标
                keyy_2 = int(match_2.group(2))  # 提取纵坐标
                print("终点横坐标:", keyx_2)
                print("终点纵坐标:", keyy_2)
                grid_widget.painting_end(keyx_2, keyy_2)
            else:
                print("坐标格式不正确")
            label_notice.setText("输入起始点生成成功！")
        # 连接按钮的点击信号
        self.button_modify.clicked.connect(on_button_click)

    # 选择图形障碍物生成窗口 单个生成
    def graph_ob(self, MainWindow, grid_widget):
        self.loginWindow_new = None
        MainWindow.setObjectName("图形障碍物")
        MainWindow.setFixedSize(300, 120)
        # 创建起点标签
        self.label_graph = QLabel("障碍物类型:", MainWindow)
        self.label_graph.setGeometry(60, 20, 80, 30)  # 设置标签位置和大小
        # 创建下拉列表
        self.combo_box = QComboBox(MainWindow)
        self.combo_box.setGeometry(130, 20, 120, 30)  # 设置下拉列表位置和大小
        self.combo_box.addItem("矩形")
        self.combo_box.addItem("圆形")
        self.combo_box.addItem("三角形")
        self.combo_box.addItem("椭圆形")
        self.combo_box.addItem("菱形")
        self.combo_box.addItem("五角形")
        # 创建生成障碍物按钮
        self.button_modify = QPushButton("生成", MainWindow)
        self.button_modify.setGeometry(90, 60, 120, 30)  # 设置按钮位置和大小
        label_notice = QLabel("", MainWindow)
        label_notice.setGeometry(90, 90, 150, 30)  # 设置标签位置和大小

        # 按钮点击事件处理函数
        def on_button_click():
            selected_index = self.combo_box.currentIndex()
            grid_widget.paint_random_one(selected_index+1)
            label_notice.setText("障碍物生成成功！")
        # 连接按钮的点击信号到处理函数
        self.button_modify.clicked.connect(on_button_click)

    def result_single(self, MainWindow, grid_widget, index, f):
        """
        结果分析窗口设计
        :param MainWindow: 当前窗体
        :param grid_widget: grid_widget
        :param index: 当前是选中的第几个文件，用于调整窗体显示的位置
        :param f: 文件的路径
        :return: None
        """
        MainWindow.setGeometry(100 + 10 * index, 100 + 10 * index, 500, 500)
        r = load_demo(f)
        canvas = r.draw_track()
        label_time = QLabel("运行时间是：" + str(r.time), MainWindow)
        label_smoothness = QLabel("路径平滑度是：" + str(r.smoothness), MainWindow)
        label_path_length = QLabel("路径长度是：" + str(r.pathlen), MainWindow)
        button_smoothness = QPushButton("展示曲率")

        def on_button_click():
            new_window = QtWidgets.QMainWindow()
            curvature = r.draw_curvature()
            new_window.setCentralWidget(curvature)
            new_window.setWindowTitle('曲率展示')
            new_window.show()
            self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中
        button_smoothness.clicked.connect(on_button_click)

        button_save = QPushButton("保存图片")

        def on_save_button_click():
            file_path, _ = QFileDialog.getSaveFileName(MainWindow, "保存图片", "",
                                                       "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                canvas.grab().save(file_path)

        button_save.clicked.connect(on_save_button_click)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        layout.addWidget(label_smoothness)
        layout.addWidget(label_path_length)
        layout.addWidget(label_time)
        layout.addWidget(button_smoothness)
        layout.addWidget(button_save)
        main_widget = QWidget(MainWindow)
        main_widget.setLayout(layout)
        MainWindow.setCentralWidget(main_widget)
        MainWindow.show()
    def map_look(self, MainWindow, grid_widget, index, f):
        """
        结果分析窗口设计
        :param MainWindow: 当前窗体
        :param grid_widget: grid_widget
        :param index: 当前是选中的第几个文件，用于调整窗体显示的位置
        :param f: 文件的路径
        :return: None
        """
        MainWindow.setGeometry(100 + 10 * index, 100 + 10 * index, 500, 500)
        r = load_look(f)
        canvas = r.draw_map()
        #button_smoothness = QPushButton("展示曲率")
        button_save = QPushButton("保存图片")

        def on_save_button_click():
            file_path, _ = QFileDialog.getSaveFileName(MainWindow, "保存图片", "",
                                                       "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                canvas.grab().save(file_path)
        button_save.clicked.connect(on_save_button_click)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        layout.addWidget(button_save)
        main_widget = QWidget(MainWindow)
        main_widget.setLayout(layout)
        MainWindow.setCentralWidget(main_widget)
        MainWindow.show()
    def result_category(self, MainWindow, grid_widget, category):
        """
        多次结果分析窗口设计
        :param MainWindow: 窗口
        :param grid_widget: grid_widget
        :param category: 多次结果类
        :return: None
        """
        MainWindow.setGeometry(100, 100, 500, 500)
        button_path_compare = QPushButton("路径叠加")
        def on_button_path_compare_click():
            self.open_result_track_compare(category)
        button_path_compare.clicked.connect(on_button_path_compare_click)
        button_average = QPushButton("求平均值")
        def on_button_average():
            self.open_result_average(category)
        button_average.clicked.connect(on_button_average)
        layout = QVBoxLayout()
        layout.addWidget(button_path_compare)
        layout.addWidget(button_average)
        main_widget = QWidget(MainWindow)
        main_widget.setLayout(layout)
        MainWindow.setCentralWidget(main_widget)
        MainWindow.show()

    def result_category_average(self, MainWindow, grid_widget, category):
        """
        多结果分析求平均值窗口设计
        :param category:
        :return:
        """
        MainWindow.setGeometry(100, 100, 500, 500)
        label_smooth = QLabel("平均平滑度是："+str(category.ave_smooth))
        label_time = QLabel("平均时间是："+str(category.ave_time))
        label_path_length = QLabel("平均路径长度是："+str(category.ave_path_length))
        textbox_name = QLineEdit("")
        textbox_name.setPlaceholderText("请输入文件名")
        button_save = QPushButton("保存结果")
        def on_button_save_click():
            dialog = QFileDialog()
            dialog.setAcceptMode(QFileDialog.AcceptSave)
            # 设置对话框标题
            dialog.setWindowTitle('保存category')
            # 设置文件过滤器
            dialog.setNameFilter('category文件 (*.txt)')
            # 设置默认文件名，包含文件类型后缀
            dialog.setDefaultSuffix('txt')
            # 打开文件对话框，并返回保存的文件路径
            file_path, _ = dialog.getSaveFileName(MainWindow, '保存category', '', 'category文件 (*.txt)')
            category.save_file(file_path, textbox_name.text())
        button_save.clicked.connect(on_button_save_click)
        layout = QVBoxLayout()
        layout.addWidget(label_time)
        layout.addWidget(label_path_length)
        layout.addWidget(label_smooth)
        layout.addWidget(textbox_name)
        layout.addWidget(button_save)
        main_widget = QWidget(MainWindow)
        main_widget.setLayout(layout)
        MainWindow.setCentralWidget(main_widget)

    def result_category_compare(self, MainWindow, grid_widget, compare):
        MainWindow.setGeometry(100, 100, 500, 500)
        button_image = QPushButton("对比结果生成图片")
        def on_button_image_click():
            dialog = QFileDialog()
            dialog.setAcceptMode(QFileDialog.AcceptSave)
            dialog.setWindowTitle('保存表格')
            dialog.setDefaultSuffix('png')
            file_path, _ = dialog.getSaveFileName(MainWindow, '保存表格', '', '(*.png)')
            compare.save_image(file_path)
        button_image.clicked.connect(on_button_image_click)
        button_csv = QPushButton("对比结果生成csv")
        def on_button_csv_clink():
            dialog = QFileDialog()
            dialog.setAcceptMode(QFileDialog.AcceptSave)
            dialog.setWindowTitle('保存表格')
            dialog.setDefaultSuffix('csv')
            file_path, _ = dialog.getSaveFileName(MainWindow, '保存表格', '', '(*.csv)')
            compare.save_csv(file_path)
        button_csv.clicked.connect(on_button_csv_clink)
        layout = QVBoxLayout()
        layout.addWidget(button_image)
        layout.addWidget(button_csv)
        main_widget = QWidget(MainWindow)
        main_widget.setLayout(layout)
        MainWindow.setCentralWidget(main_widget)

    def setupUi(self, MainWindow, grid_widget):
        self.loginWindow_new = None
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 850)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./img/logo.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1100, 850))
        MainWindow.setMaximumSize(QtCore.QSize(1100, 850))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(130, 425, 75, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(grid_widget.clear_map)
        self.text_result = QtWidgets.QTextBrowser(self.centralwidget)
        self.text_result.setGeometry(QtCore.QRect(40, 500, 921, 251))
        self.text_result.setObjectName("text_result")
        # 获取图形按钮
        self.pushButton_new = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_new.setGeometry(QtCore.QRect(40, 425, 75, 25))
        self.pushButton_new.setObjectName("pushButton_new")
        self.grid_widget = grid_widget
        self.pushButton_new.clicked.connect(self.grid_widget.get_obs_vertices)
        # 地图栅格化按钮
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(220, 425, 75, 25))
        self.pushButton_3.setObjectName("pushButton_3")
        # 地图栅格方法
        self.pushButton_3.clicked.connect(self.grid_widget.rasterize_map)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(40, 20, 921, 450))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setObjectName("layout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1006, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        # 添加新的工具栏
        self.menu_ob = QtWidgets.QMenu(self.menubar)
        self.menu_ob.setObjectName("menu_ob")
        self.menu_startAndEnd = QtWidgets.QMenu(self.menubar)
        self.menu_startAndEnd.setObjectName("menu_startAndEnd")
        self.menu_map = QtWidgets.QMenu(self.menubar)
        self.menu_map.setObjectName("menu_map")
        self.menu_plan = QtWidgets.QMenu(self.menubar)
        self.menu_plan.setObjectName("menu_plan")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolBar)
        self.toolBar.setMovable(False)  # 设置工具栏为不可移动
        self.actionLoadTest = QtWidgets.QAction(MainWindow)
        self.actionLoadTest.setObjectName("actionLoadTest")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.loginOut)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(grid_widget.open_map)
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(180, 455, 80, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.open_randomOb)
        # 随机起始点按钮
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(550, 455, 80, 23))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.clicked.connect(grid_widget.generateRandomStart)
        # 分析规划按钮
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(640, 455, 80, 23))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.clicked.connect(self.open_start_path) # 方法
        # 单路径规划
        self.pushButton_plan = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_plan.setGeometry(QtCore.QRect(640, 455, 80, 23))
        self.pushButton_plan.setObjectName("pushButton_plan")
        self.pushButton_plan.clicked.connect(self.select_method) # 方法
        # 地图调整
        self.pushButton_modify_map = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_modify_map.setGeometry(QtCore.QRect(640, 455, 80, 23))
        self.pushButton_modify_map.setObjectName("pushButton_modify_map")
        self.pushButton_modify_map.clicked.connect(self.modify_map)  # 方法
        #预览地图
        self.pushButton_look_map = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_look_map.setGeometry(QtCore.QRect(640, 455, 80, 23))
        self.pushButton_look_map.setObjectName("pushButton_look_map")
        self.pushButton_look_map.clicked.connect(self.look_map)  # 方法
        # 输入起始点按钮
        self.pushButton_input_startAndEnd = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_input_startAndEnd.setGeometry(QtCore.QRect(640, 455, 80, 23))
        self.pushButton_input_startAndEnd.setObjectName("pushButton_input_startAndEnd")
        self.pushButton_input_startAndEnd.clicked.connect(self.startAndEnd)  # 方法
        self.actionCreate = QtWidgets.QAction(MainWindow)

        self.actionCreate.setObjectName("actionCreate")
        # 点击菜单连接方法
        self.actionCreate.triggered.connect(self.openNewWindow)
        # 点击菜单退出方法
        self.actionExit.triggered.connect(self.loginOut)
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        # 保存地图的方法
        self.actionSave.triggered.connect(grid_widget.save_map)
        self.createArithmetic = QtWidgets.QAction(MainWindow)
        self.createArithmetic.setObjectName("createArithmetic")
        self.createArithmetic.triggered.connect(self.openArithmeticList)
        self.actionVersion = QtWidgets.QAction(MainWindow)
        self.actionVersion.setObjectName("actionVersion")
        # 版本信息弹出框链接
        self.actionVersion.triggered.connect(self.versionInformation)
        self.actionhelp = QtWidgets.QAction(MainWindow)
        self.actionhelp.setObjectName("actionhelp")
        # 帮助手册弹出框链接
        self.actionhelp.triggered.connect(self.helpInfo)
        self.actionmodel = QtWidgets.QAction(MainWindow)
        self.actionmodel.setObjectName("actionmodel")
        # 生成单个图形障碍物可控大小
        self.pushButton_paint_rand = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_paint_rand.setGeometry(QtCore.QRect(730, 454, 80, 25))
        self.pushButton_paint_rand.setObjectName("pushButton_paint_rand")
        self.pushButton_paint_rand.clicked.connect(self.select_graph)
        # 参数障碍物按钮
        self.pushButton_ob = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_ob.setGeometry(QtCore.QRect(730, 454, 80, 25))
        self.pushButton_ob.setObjectName("pushButton_ob")
        self.pushButton_ob.clicked.connect(self.open_modifyOb)
        # 动态障碍物按钮
        self.pushButton_dynamic_ob = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_dynamic_ob.setGeometry(QtCore.QRect(730, 454, 80, 25))  # 设置按钮的位置和大小
        self.pushButton_dynamic_ob.setObjectName("pushButton_dynamic_ob")
        self.pushButton_dynamic_ob.setText("动态障碍物")  # 设置按钮的文字
        self.pushButton_dynamic_ob.clicked.connect(self.handle_dynamic_ob)  # 绑定点击事件
        #self.pushButton_dynamic_ob.setStyleSheet("background-color: yellow;")  # 设置背景色以便观察
        #self.pushButton_dynamic_ob.setVisible(True)  # 确保按钮可见
         # 新增：实时模拟按钮
        self.btn_realtime_sim = QtWidgets.QPushButton(self.centralwidget)
        self.btn_realtime_sim.setGeometry(QtCore.QRect(730, 454, 80, 25))  # 设置按钮的位置和大小
        self.btn_realtime_sim.setObjectName("btn_realtime_sim")
        self.btn_realtime_sim.setText("实时航行")  # 设置按钮的文字
        self.btn_realtime_sim.clicked.connect(self.start_realtime_simulation)  # 绑定点击事件
        
        # 新增：局部路径规划算法选择框
        self.label_local_algorithm = QtWidgets.QLabel(self.centralwidget)
        self.label_local_algorithm.setGeometry(QtCore.QRect(820, 454, 100, 25))
        self.label_local_algorithm.setObjectName("label_local_algorithm")
        self.label_local_algorithm.setText("局部算法:")
        
        self.combo_local_algorithm = QtWidgets.QComboBox(self.centralwidget)
        self.combo_local_algorithm.setGeometry(QtCore.QRect(920, 454, 120, 25))
        self.combo_local_algorithm.setObjectName("combo_local_algorithm")
        self.combo_local_algorithm.addItem("无局部算法")
        self.combo_local_algorithm.addItem("APF算法")
        self.combo_local_algorithm.addItem("DWA算法")  # 预留，待实现
        self.combo_local_algorithm.addItem("人工势场法")
        self.combo_local_algorithm.addItem("动态窗口法")  # 预留，待实现
        self.combo_local_algorithm.setCurrentIndex(0)  # 默认选择无局部算法
        # self.pushButton_dynamic_ob.setStyleSheet("background-color: yellow;")  # 设置背景色以便观察
        # self.pushButton_dynamic_ob.setVisible(True)  # 确保按钮可见

        # 下载地图模板
        self.actionArithmeticList = QtWidgets.QAction(MainWindow)
        self.actionArithmeticList.setObjectName("actionArithmeticList")
        self.actionArithmeticList.triggered.connect(self.openArithmeticList)
        # 随机障碍物
        self.action_random = QtWidgets.QAction(MainWindow)
        self.action_random.setObjectName("action_random")
        self.action_random.triggered.connect(self.open_randomOb)
        # 参数障碍物
        self.action_canshu = QtWidgets.QAction(MainWindow)
        self.action_canshu.setObjectName("action_canshu")
        self.action_canshu.triggered.connect(self.open_modifyOb)
        # 图形障碍物
        self.action_graph = QtWidgets.QAction(MainWindow)
        self.action_graph.setObjectName("action_graph")
        self.action_graph.triggered.connect(self.select_graph)
        # 随机起始点
        self.action_randomPoint = QtWidgets.QAction(MainWindow)
        self.action_randomPoint.setObjectName("action_randomPoint")
        self.action_randomPoint.triggered.connect(grid_widget.generateRandomStart)
        # 输入起始点
        self.action_input = QtWidgets.QAction(MainWindow)
        self.action_input.setObjectName("action_input")
        self.action_input.triggered.connect(self.startAndEnd)
        # 清空地图
        self.action_empty = QtWidgets.QAction(MainWindow)
        self.action_empty.setObjectName("action_empty")
        self.action_empty.triggered.connect(grid_widget.clear_map)
        # 调整地图
        self.action_mapModify = QtWidgets.QAction(MainWindow)
        self.action_mapModify.setObjectName("action_mapModify")
        self.action_mapModify.triggered.connect(self.modify_map)
        # 路径规划
        self.action_plan = QtWidgets.QAction(MainWindow)
        self.action_plan.setObjectName("action_plan")
        self.action_plan.triggered.connect(self.select_method)
        # 分析规划
        self.action_analyse = QtWidgets.QAction(MainWindow)
        self.action_analyse.setObjectName("action_analyse")
        self.action_analyse.triggered.connect(self.open_start_path)
        # 获取图形
        self.action_get = QtWidgets.QAction(MainWindow)
        self.action_get.setObjectName("action_get")
        self.action_get.triggered.connect(self.grid_widget.get_obs_vertices)
        # 栅格化地图
        self.action_rasterization = QtWidgets.QAction(MainWindow)
        self.action_rasterization.setObjectName("action_rasterization")
        self.action_rasterization.triggered.connect(self.grid_widget.rasterize_map)
        self.menu.addAction(self.actionCreate)
        self.menu.addAction(self.actionOpen)
        self.menu.addAction(self.actionSave)
        self.menu.addAction(self.createArithmetic)
        self.menu.addAction(self.actionmodel)
        self.menu.addAction(self.actionArithmeticList)
        self.menu_5.addAction(self.actionExit)
        self.menu_5.addAction(self.actionVersion)
        self.menu_5.addAction(self.actionhelp)
        self.menu_ob.addAction(self.action_random)
        self.menu_ob.addAction(self.action_canshu)
        self.menu_ob.addAction(self.action_graph)
        self.menu_startAndEnd.addAction(self.action_randomPoint)
        self.menu_startAndEnd.addAction(self.action_input)
        self.menu_map.addAction(self.action_empty)
        self.menu_map.addAction(self.action_mapModify)
        self.menu_plan.addAction(self.action_plan)
        self.menu_plan.addAction(self.action_analyse)
        self.menu_plan.addAction(self.action_get)
        self.menu_plan.addAction(self.action_rasterization)

        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_ob.menuAction())
        self.menubar.addAction(self.menu_startAndEnd.menuAction())
        self.menubar.addAction(self.menu_map.menuAction())
        self.menubar.addAction(self.menu_plan.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.toolBar.addAction(self.actionCreate)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionmodel)
        self.toolBar.addAction(self.actionExit)

        self.toolBar.addAction(self.actionCreate)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionmodel)
        self.toolBar.addAction(self.actionhelp)
        self.toolBar.addAction(self.actionArithmeticList)
        self.toolBar.addAction(self.actionExit)
        # 侧边栏
        self.sideToolBar = QtWidgets.QToolBar(MainWindow)
        self.sideToolBar.setObjectName("sideToolBar")
        # self.sideToolBar.setMovable(False)  # 设置工具栏为不可移动
        MainWindow.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, self.sideToolBar)
        self.sideToolBar.addWidget(self.pushButton_plan)
        self.sideToolBar.addWidget(self.pushButton_7)
        self.sideToolBar.addWidget(self.pushButton_new)
        self.sideToolBar.addWidget(self.pushButton_3)
        self.sideToolBar.addWidget(self.pushButton)
        self.sideToolBar.addWidget(self.pushButton_6)
        self.sideToolBar.addWidget(self.pushButton_input_startAndEnd)
        self.sideToolBar.addWidget(self.pushButton_paint_rand)
        self.sideToolBar.addWidget(self.pushButton_ob)
        self.sideToolBar.addWidget(self.pushButton_dynamic_ob)
        self.sideToolBar.addWidget(self.btn_realtime_sim)  # 添加开始实时航行按钮
        self.sideToolBar.addWidget(self.label_local_algorithm)
        self.sideToolBar.addWidget(self.combo_local_algorithm)
        self.sideToolBar.addWidget(self.pushButton_5)
        self.sideToolBar.addWidget(self.pushButton_modify_map)
        self.sideToolBar.addWidget(self.pushButton_look_map)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def handle_dynamic_ob(self):
        # 创建一个新的窗口
        dynamic_ob_window = QtWidgets.QMainWindow()
        ui.DynamicObstaclesUI(dynamic_ob_window, self.grid_widget)  # 专门处理动态障碍物的UI
        # 设置窗口标题
        dynamic_ob_window.setWindowTitle('动态障碍物')
        # 初始化动态障碍物界面

        #ui.setupUi(dynamic_ob_window, self.grid_widget)
        # 显示新窗口
        dynamic_ob_window.show()
        # 将窗口实例保存到窗口列表中，避免被垃圾回收
        self.windows.append(dynamic_ob_window)

    # 图形障碍物窗口
    def select_graph(self):
        new_window = QtWidgets.QMainWindow()
        ui.graph_ob(new_window, self.grid_widget)
        new_window.setWindowTitle('图形障碍物')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中
    # 随机障碍物窗口
    def open_randomOb(self):
        new_window = QtWidgets.QMainWindow()
        ui.random_ob(new_window, self.grid_widget)
        new_window.setWindowTitle('随机障碍物')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中

    # 随机障碍物窗口
    def startAndEnd(self):
        new_window = QtWidgets.QMainWindow()
        ui.input_startAndEnd(new_window, self.grid_widget)
        new_window.setWindowTitle('输入起始点')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中
    # 随机障碍物窗口
    def modify_map(self):
        new_window = QtWidgets.QMainWindow()
        ui.map(new_window, self.grid_widget)
        new_window.setWindowTitle('随机障碍物')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中

    def look_map(self):

        # 创建文件对话框
        dialog = QFileDialog()
        # 设置对话框标题
        dialog.setWindowTitle('打开结果文件')
        # 设置文件过滤器
        dialog.setNameFilter('Text Files (*.txt)')
        # 设置默认文件名，包含文件类型后缀
        # dialog.setDefaultSuffix('txt')
        files, _ = dialog.getOpenFileNames()
        for index, f in enumerate(files):  # 循环选中的所有文件
            new_window = QtWidgets.QMainWindow()
            ui.map_look(new_window, self.grid_widget, index, f)
            new_window.setWindowTitle('地图预览')
            new_window.show()
            self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中
    # 路径规划选择算法窗口
    def select_method(self):
        new_window = QtWidgets.QMainWindow()
        ui.select_arithmetic(new_window, self.grid_widget)
        new_window.setWindowTitle('仿真算法')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中
    # 打开参数障碍物窗口
    def open_modifyOb(self):
        new_window = QtWidgets.QMainWindow()
        ui.modify_obstacles(new_window, self.grid_widget)
        new_window.setWindowTitle('参数障碍物')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中

        # 打开规划算法窗口
    def open_start_path(self):
        new_window = QtWidgets.QMainWindow()
        ui.start_path(new_window, self.grid_widget)
        new_window.setWindowTitle('规划算法')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中

    def open_result_single(self, index, f):
        """
        打开单路径分析窗口。在分析路径的窗口里打开。
        :param index:当前的文件是选中的第几个文件，用于调整窗口的位置
        :param f:文件的路径
        :return:None
        """
        new_window = QtWidgets.QMainWindow()
        ui.result_single(new_window, self.grid_widget, index, f)
        new_window.setWindowTitle('单路径分析')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中

    def open_result_category(self, category):
        """
        打开多路径分析窗口。在分析路径的窗口打开。
        :return:
        """
        new_window = QtWidgets.QMainWindow()
        ui.result_category(new_window, self.grid_widget, category)
        new_window.setWindowTitle('多路径分析')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中

    def open_result_track_compare(self, category):
        """
        打开多路径分析中路径叠加的窗口
        :category:多结果
        :return:None
        """
        new_window = QtWidgets.QMainWindow()
        canvas = category.track_compare()
        new_window.setCentralWidget(canvas)
        new_window.setWindowTitle('多路径分析')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中

    def open_result_average(self, category):
        """
        打开多路径分析中计算平均值的窗口
        :param category: 多结果
        :return: None
        """
        new_window = QtWidgets.QMainWindow()
        ui.result_category_average(new_window, self.grid_widget, category)
        new_window.setWindowTitle('多路径分析')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中
    def open_result_category_compare(self, compare):
        """
        打开多算法比较信息的窗口
        :param compare: 比较信息,包含多个category
        :return: None
        """
        new_window = QtWidgets.QMainWindow()
        ui.result_category_compare(new_window, self.grid_widget, compare)
        new_window.setWindowTitle('多路径分析')
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中

    # 文本框输出提示信息
    def printf(self, msg, x=None, y=None):
        if x is None and y is None:
            self.text_result.append("%s" % msg)
        else:
            self.text_result.append("%s(%d:%d)" % (msg, x, y))

    # 创建新窗口方法
    def openNewWindow(self):
        new_window = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        grid_widget = GridWidget(ui)
        pw = PygameWidget(ui)
        ui.setupUi(new_window, pw)

        new_window.setWindowTitle('基于Pygame的路径规划算法仿真平台')
        # 添加地图
        ui.layout.addWidget(pw)
        new_window.show()
        self.windows.append(new_window)  # 将新创建的窗口实例添加到列表中

    # 退出登录
    def loginOut(self):
        self.MainWindow.hide()
        # TODO 退出功能的实现，退出时问用户是否保存地图数据

    # 版本信息弹出框 TODO 需要更改
    def versionInformation(self):
        message_box = QMessageBox()
        message_box.setWindowTitle("基于Pygame的路径规划算法仿真平台V1.0")
        message_box.setText(
            "路径规划算法仿真平台可以帮助实际应用中的路径规划问题，如无人车、无人机、物流配送等领域的规划和优化。通过仿真平台，可以提前预测和分析路径规划算法在实际环境中的表现，从而减少实际试验的成本和风险。同时，仿真平台还可以为实际应用中的路径规划算法提供实时的优化和调整，以满足不同场景和需求的要求。"
            "路径规划仿真平台V1.0版本是指一种软件工具，可以模拟路径，通过路径规划算法计算最优路径，并可视化显示路径和相关参数。用户可以通过输入环境地图和调整参数等方式，对仿真平台进行操作和控制。")
        message_box.setIcon(QMessageBox.Information)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.button(QMessageBox.Ok).setFixedSize(100, 40)  # 设置按钮尺寸
        message_box.setFixedSize(800, 800)  # 设置提示框尺寸

        message_box.exec()

    # 帮助信息提示 TODO 需要更改
    def helpInfo(self):
        message_box = QMessageBox()
        message_box.setWindowTitle("基于Pygame的路径规划算法仿真平台V1.0帮助手册")
        message_box.setText(
            "路径规划仿真平台V1.0版本是指一种软件工具，可以模拟路径，通过路径规划算法计算最优路径，并可视化显示路径和相关参数。用户可以通过输入环境地图和调整参数等方式，对仿真平台进行操作和控制。"
            "仿真平台的使用：本平台使用手动设置起点、终点与障碍点的位置，通过鼠标左键设置障碍点，鼠标右键点击第一次为红色起点点击第二次为绿色终点，地图是可以实时清空和刷新的，只要用户对地图进行改变"
            "平台就会自动刷新，并且本平台采用可以手动调整地图的粒度大小的，但是建议用户采用的地图粒度大小在10-50区间为最佳效果。")
        message_box.setIcon(QMessageBox.Information)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.button(QMessageBox.Ok).setFixedSize(100, 40)  # 设置按钮尺寸
        message_box.setFixedSize(800, 800)  # 设置提示框尺寸
        message_box.exec()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于Pygame的路径规划算法仿真平台"))
        self.pushButton.setText(_translate("MainWindow", "清空地图"))
        self.text_result.setPlaceholderText(_translate("MainWindow", "输出路径规划结果"))
        self.text_result.append(
            "欢迎来到基于Pygame的路径规划算法仿真平台，以下是路径规划的结果仅供大家参考（右键按第一次是起点第二次是终点，左键设置起点）：")
        self.text_result.append("红色为起点、绿色为终点、黑色为障碍点，平台具体方法请点击帮助手册查看！")
        self.pushButton_new.setText(_translate("MainWindow", "获取图形"))
        self.pushButton_3.setText(_translate("MainWindow", "地图栅格化"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_5.setTitle(_translate("MainWindow", "关于"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionLoadTest.setText(_translate("MainWindow", "压力测试"))
        self.actionExit.setText(_translate("MainWindow", "退出"))
        self.actionOpen.setText(_translate("MainWindow", "打开文件"))
        self.actionCreate.setText(_translate("MainWindow", "新建窗口"))
        self.actionSave.setText(_translate("MainWindow", "保存地图"))
        self.createArithmetic.setText(_translate("MainWindow", "新增算法"))
        self.actionVersion.setText(_translate("MainWindow", "版本信息"))
        self.actionhelp.setText(_translate("MainWindow", "帮助手册"))
        self.actionmodel.setText(_translate("MainWindow", "下载地图模板"))
        self.pushButton_plan.setText(_translate("MainWindow", "路径规划"))
        self.actionArithmeticList.setText(_translate("MainWindow", "算法列表"))
        self.pushButton_5.setText(_translate("MainWindow", "随机障碍物"))
        self.pushButton_6.setText(_translate("MainWindow", "随机起始点"))
        self.pushButton_7.setText(_translate("MainWindow", "分析规划"))
        self.pushButton_paint_rand.setText(_translate("MainWindow", "图形障碍物"))
        self.pushButton_ob.setText(_translate("MainWindow", "参数障碍物"))
        self.pushButton_modify_map.setText(_translate("MainWindow", "地图调整"))
        self.pushButton_look_map.setText(_translate("MainWindow", "预览地图"))
        self.pushButton_input_startAndEnd.setText(_translate("MainWindow", "输入起始点"))
        self.menu_ob.setTitle(_translate("MainWindow", "障碍物设置"))
        self.menu_startAndEnd.setTitle(_translate("MainWindow", "起始点设置"))
        self.menu_map.setTitle(_translate("MainWindow", "地图设置"))
        self.menu_plan.setTitle(_translate("MainWindow", "路径规划"))
        self.action_random.setText(_translate("MainWindow", "随机障碍物"))
        self.action_canshu.setText(_translate("MainWindow", "参数障碍物"))
        self.action_graph.setText(_translate("MainWindow", "图形障碍物"))
        self.action_randomPoint.setText(_translate("MainWindow", "随机起始点"))
        self.action_input.setText(_translate("MainWindow", "输入起始点"))
        self.action_empty.setText(_translate("MainWindow", "清空地图"))
        self.action_mapModify.setText(_translate("MainWindow", "调整地图"))
        self.action_plan.setText(_translate("MainWindow", "路径规划"))
        self.action_analyse.setText(_translate("MainWindow", "分析规划"))
        self.action_get.setText(_translate("MainWindow", "获取图形"))
        self.action_rasterization.setText(_translate("MainWindow", "栅格化"))
    # 路径规划
    def start_realtime_simulation(self):
        """启动实时模拟"""
        self.grid_widget.plan_surface.fill((255, 255, 255))
        
        # 获取当前选择的局部算法
        local_algorithm = self.combo_local_algorithm.currentText()
        
        # 设置局部算法到运动模拟器
        if hasattr(self.grid_widget, 'motion_simulator') and self.grid_widget.motion_simulator:
            self.grid_widget.motion_simulator.set_local_algorithm(local_algorithm)
        
        self.grid_widget.start_realtime_simulation()
    def startPath(self):
        self.grid_widget.plan_surface.fill((255, 255, 255))
        if not self.grid_widget.obstacles or not self.grid_widget.start_point or not self.grid_widget.end_point:
            self.printf("未设置障碍物或起始点或终点")
        # 根据不同的算法
        else:
            if self.combo_arithmetic.currentText() == "Astar":
                self.printf("启动A星算法！！！")
                self.grid_widget.startAstar()
            elif self.combo_arithmetic.currentText() == "RRT":
                self.printf("启动RRT算法！！！")
                self.grid_widget.start_rrt()
            elif self.combo_arithmetic.currentText() == "APF":
                self.printf("启动APF算法！！！")
                self.grid_widget.startApf()
            elif self.combo_arithmetic.currentText() == "APFRRT":
                self.printf("启动APF-RRT算法！！！")
                self.grid_widget.startApfRrt()
            elif self.combo_arithmetic.currentText() == "PRM":
                self.printf("启动PRM算法！！！")
                self.grid_widget.startPRm()

    def ori_end_input(self):  # 输入起始点终点函数
        coordinate = self.text_input.text()
        print(coordinate)
        pattern = r"\((\d+),(\d+)\)"  # 匹配坐标的正则表达式模式
        match = re.match(pattern, coordinate)
        print(match)
        if match:
            keyx = int(match.group(1))  # 提取横坐标
            keyy = int(match.group(2))  # 提取纵坐标
            print("起点横坐标:", keyx)
            print("起点纵坐标:", keyy)
            pw.painting_ori(keyx, keyy)  # 目前执行到这里程序就结束了，应该是调用有问题
        else:
            print("坐标格式不正确")
        coordinate_2 = self.text_input_2.text()
        # print(coordinate_2)
        pattern_2 = r"\((\d+),(\d+)\)"  # 匹配坐标的正则表达式模式
        match_2 = re.match(pattern, coordinate_2)
        if match_2:
            # global keyx_2, keyy_2
            keyx_2 = int(match_2.group(1))  # 提取横坐标
            keyy_2 = int(match_2.group(2))  # 提取纵坐标
            print("终点横坐标:", keyx_2)
            print("终点纵坐标:", keyy_2)
            pw.painting_end(keyx_2, keyy_2)
        else:
            print("坐标格式不正确")

    def openArithmeticList(self):
        self.algorithm_list = AlgorithmList()
        self.algorithm_list.show()




if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    mainWindow = QtWidgets.QMainWindow()
    grid_widget = GridWidget(ui)
    pw = PygameWidget(ui)
    ui.setupUi(mainWindow, pw)
    # 添加地图
    ui.layout.addWidget(pw)
    mainWindow.show()
    sys.exit(app.exec())
