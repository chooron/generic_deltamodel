import random
import time

class Cat:
    """
    一只猫，拥有：
    - 品种（breed）
    - 颜色（color）
    - 名字（name）
    以及三种基本行为：
    - eat      吃东西
    - sleep    睡觉
    - play     玩耍
    """

    def __init__(self, name: str, breed: str, color: str):
        """构造函数：给猫起名字、品种、颜色，并初始化状态值"""
        self.name   = name
        self.breed  = breed
        self.color  = color
        # 下面 3 个是内部状态，默认从 0~100 计数
        self.energy = 100   # 精力
        self.hunger = 0     # 饥饿度（0 表示饱，100 表示饿坏了）
        self.is_asleep = False   # 是否正在睡觉

    # --------------------
    # 行为：吃东西
    # --------------------
    def eat(self, food: str = "猫粮") -> str:
        """让猫吃东西，会减饥饿、加精力"""
        if self.is_asleep:
            return f"{self.name} 正在睡觉，吃不到 {food}！"
        self.hunger = max(0, self.hunger - 40)          # 饥饿值下降
        self.energy = min(100, self.energy + 20)        # 精力值上升
        return f"{self.name} 吃了 {food}，饥饿值降到 {self.hunger}，精力提升到 {self.energy}。"

    # --------------------
    # 行为：睡觉
    # --------------------
    def sleep(self, seconds: float = 1.0) -> str:
        """让猫睡觉，会恢复精力"""
        if self.is_asleep:
            return f"{self.name} 已经睡着了，不要吵它。"
        self.is_asleep = True
        # 假装睡觉需要花时间
        time.sleep(seconds)
        recover = random.randint(30, 50)
        self.energy = min(100, self.energy + recover)
        self.is_asleep = False
        return f"{self.name} 睡了一觉，精力回复了 {recover} 点，现在是 {self.energy}。"

    # --------------------
    # 行为：玩耍
    # --------------------
    def play(self, toy: str = "毛线球") -> str:
        """让猫玩耍，会消耗精力、增加饥饿"""
        if self.is_asleep:
            return f"{self.name} 正在睡觉，不能玩 {toy}！"
        if self.energy < 20:
            return f"{self.name} 太累了，需要先睡觉或吃东西。"
        self.energy -= 30
        self.hunger = min(100, self.hunger + 20)
        return f"{self.name} 玩了 {toy}，精力降到 {self.energy}，饥饿值升到 {self.hunger}。"

    # --------------------
    # 额外：打印当前状态
    # --------------------
    def status(self) -> str:
        """返回猫的当前状态，方便查看"""
        awake = "睡觉中" if self.is_asleep else "醒着"
        return (f"{self.color} 色的 {self.breed} 猫 {self.name}，"
                f"当前精力 {self.energy}，饥饿值 {self.hunger}，状态：{awake}。")


# --------------------
# 演示用法
# --------------------
if __name__ == "__main__":
    tom = Cat("Tom", "英短", "蓝灰")
    print(tom.status())      # 初始状态
    print(tom.play())        # 玩一玩
    print(tom.eat("鱼"))     # 吃东西
    print(tom.sleep(0.5))    # 睡觉（时间缩短，演示用）
    print(tom.status())      # 再看一眼