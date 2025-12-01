import arrow

def is_number(s):
    try:
        float(s)  # 尝试转换为 float（可以处理整数和浮点数）
        return True
    except ValueError:
        return False

# # 测试
# print(is_number("123"))      # True (整数)
# print(is_number("pi"))      # False
# print(is_number("-pi"))      # False
# print(is_number("123.45"))   # True (浮点数)
# print(is_number("-123.45"))  # True (负数)
# print(is_number("1e5"))      # True (科学计数法)
# print(is_number("abc"))      # False (非数字)
# print(is_number("123abc"))   # False (部分数字)


def get_animals_name():
    animals = [
        "Dog",
        "Cat",
        "Cow",
        "Horse",
        "Sheep",
        "Chicken",
        "Duck",
        "Pig",
        "Rabbit",
        "Mouse",
        "Lion",
        "Tiger",
        "Bear",
        "Fox",
        "Wolf",
        "Elephant",
        "Giraffe",
        "Zebra",
        "Monkey",
        "Gorilla",
        "Kangaroo",
        "Squirrel",
        "Turtle",
        "Frog",
        "Snake",
        "Crocodile",
        "Whale",
        "Shark",
        "Dolphin",
        "Penguin",
        "Parrot"
    ]
    date = arrow.now().format('DD')
    index = int(date) - 1
    return animals[index]
