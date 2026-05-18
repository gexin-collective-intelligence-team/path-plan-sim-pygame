class point:  # 点类（每一个唯一坐标只有对应的一个实例）
    _list = []  # 储存所有的point类实例
    _tag = True  # 标记最新创建的实例是否为_list中的已有的实例，True表示不是已有实例

    def __new__(cls, x, y):  # 重写new方法实现对于同样的坐标只有唯一的一个实例
        return super(point, cls).__new__(cls)

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.father = None

    @classmethod
    def clear(cls):  # clear方法，每次搜索结束后，将所有点数据清除，以便进行下一次搜索的时候点数据不会冲突。
        point._list = []
        point._tag = True

    def __eq__(self, T):  # 重写==运算以便实现point类的in运算
        if type(self) == type(T):
            return (self.x, self.y) == (T.x, T.y)
        else:
            return False

    def __str__(self):
        return '(%d,%d)' % (self.x, self.y)
