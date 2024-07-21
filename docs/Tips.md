# Tips

## Additional Libraries: torch_scatter torch_sparse torch_cluster

[https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#additional-libraries](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#additional-libraries)

- Get PyTorch and CUDA version on your computer

```python
import torch

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda_version)

# 2.0.0
# 11.8
```

- Install the specific PyTorch and CUDA versions respectively

```cmd
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Python: self & cls

`cls`是类方法中的参数, 用于指代类自身; 而`self`是实例方法中的参数, 用于指代实例对象

- 类方法 `class method`: 用于操作类层面的数据和行为
- 实例方法 `instance method`: 用于操作实例层面的数据和行为

```python
class Car(object):
    def __init__(self, brand, color):
        self.brand = brand
        self.color = color

    # instance method
    def show_info(self):
        print(f"This is a {self.color} {self.brand} car.")

    @classmethod
    def show_info_class(cls, brand, color):
        car = cls(brand, color)  # 创建实例对象
        car.show_info()  # 调用实例方法

# 1. 类方法
Car.show_info_class("BMW", "blue")

# 2. 实例方法
car = Car("BMW", "blue")
car.show_info()
```

## 装饰器: @classmethod, @staticmethod, @property

### @classmethod

在使用的时候, 会将类本身作为第一个参数 `cls` 传递给类方法

```python
class Web(object):

    name = "Python_Web"

    def __init__(self):
        self.desc = "实例属性, 不共享"

    def norm_method(self):
        """普通方法"""
        print('普通方法被调用!')

    @staticmethod
    def foo_staticmethod():
        print('静态方法被调用!')

    @classmethod
    def foo_classmethod_other(cls):
        print('另外一个类方法被调用!')

    # 类方法, 第一个参数为cls, 代表类本身
    @classmethod
    def foo_classmethod(cls):
        # 1. 调用静态变量
        print(cls.name)
        print(Web.name)

        # 2. 调用其他类方法
        cls.foo_classmethod_other()

        # 3. 调用静态方法
        cls.foo_staticmethod()

        # 4. 要调用实例属性, 必须使用cls实例化一个对象, 然后再去引用
        print(cls().desc)

        # 5. 要调用普通方法, 必须使用cls实例化一个对象, 然后再去引用
        cls().norm_method()


if __name__ == '__main__':
    # 使用类名去调用类方法
    Web.foo_classmethod()
```

### @staticmethid

```python
class Web(object):
    # 静态变量（类变量）
    name = "Python_Web"

    def __init__(self):
        self.desc = "实例属性, 不共享"

    def norm_method(self):
        """普通方法"""
        print('普通方法被调用!')

    @classmethod
    def foo_classmethod_other(cls):
        print('类方法被调用!')

    @staticmethod
    def foo_staticmethod_other():
        print('另外一个静态方法被调用!')

    @staticmethod
    def foo_staticmethod():
        """静态方法"""
        # 1. 调用其他静态方法
        print(Web.foo_staticmethod_other()) 

        # 2. 调用类方法
        print(Web.foo_classmethod_other())

        # 3. 引用静态变量
        print(Web.name)

        # 4. 调用普通方法, 访问实例属性
        # 必须通过实例对象去引用, 不能直接使用类名去访问
        instance = Web()
        print(instance.desc)
        instance.norm_method()


if __name__ == "__name__":
    # 推荐: 直接使用类名+方法名调用
    Web.foo_staticmethod()
    # 不推荐: 实例化一个类对象, 通过这个对象去调用静态方法
    instance = Web()
    instance.foo_staticmethod()  
```

### @property

负责把一个方法变成属性调用

```python
class Student(object):

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
```

```shell
>>> s = Student()
>>> s.score = 60 # OK, 实际转化为s.set_score(60)
>>> s.score # OK, 实际转化为s.get_score()
60
>>> s.score = 9999
Traceback (most recent call last):
  ...
ValueError: score must between 0 ~ 100!
```

## 打印运行过程信息

```python
print(f"\n{'-' * 20}    Start    {'-' * 20}\n")
print(f"\n{'-' * 20}    End    {'-' * 20}\n")
```

## 常用函数

- `isinstance(name, type)`: Return whether an object is an instance of a class or of a subclass thereof.
- `__getitem__` & `__setitem__`: 创建自定义的字典类, 并对其行为进行个性化定制
- `getattr()` & `setattr()`: Get or set a named attribute from an object
- `vars()`: 函数返回对象object的属性和属性值的字典对象, Without arguments, equivalent to `locals()`, With an argument, equivalent to `object.__dict__`

## 断言: assert

用于判断一个表达式, 在表达式条件为 `false` 的时候触发异常, 直接返回错误, 而不必等待程序运行后出现崩溃的情况

```python
assert expression [, arguments]

# 等价于
if not expression:
    raise AssertionError(arguments)

# Example:
assert self.reusable == False, "self.reusable == True Unsupported currently!"
```

## Optional 类型提示

`Optional` 类型提示是用于在函数参数或返回值中标记为`可选的类型`, 这可以帮助开发者更好地理解代码的含义, 并提高代码的可读性和可维护性

– `Optional` 类型提示只能用在参数或返回值上, 不能用在变量声明上
– 对于可选参数, 在使用之前需要进行 `None` 判断, 以避免可能的异常情况
– 当使用 `Optional` 类型提示时, 编译器不会强制要求传入的参数必须为指定类型或 `None`, 仍然可以传入其他类型的值
– `Optional` 类型提示只是一种标记, 并不会改变代码的行为或执行时的结果

```python
from typing import Optional

def greet(name: Optional[str]) -> str:
    if name is None:
        return 'Hello!'
    else:
        return f'Hello, {name}!'

print(greet(None))  # 输出：Hello!
print(greet('Alice'))  # 输出：Hello, Alice!
```

## @cached_property

它将类的方法转换为一个属性, 该属性的值只计算一次, 然后缓存为普通属性

```python
from functools import cached_property

class DataSet:

    def __init__(self, sequence_of_numbers):
        self._data = tuple(sequence_of_numbers)

    @cached_property
    def stdev(self):
        return statistics.stdev(self._data)
```

类 `DataSet` 的方法 `DataSet.stdev()` 在生命周期内变成了属性 `DataSet.stdev`

缓存的 `cached_property` 装饰器仅在查找时运行, 并且仅在同名属性不存在时运行. 当它运行时, `cached_property` 会写入具有相同名称的属性. 后续的属性读取和写入优先于缓存的 `cached_property` 方法, 其工作方式与普通属性类似.

缓存的值可通过删除该属性来清空.  这允许 `cached_property` 方法再次运行

## ValueError: attempted relative import beyond top-level package

导致这个问题的原因: `主模块(程序入口, 通常指: main.py)` 所在同级包的子模块在使用相对导入时引用了主模块所在包.

因为主模块所在包不会被python解释器视为`package`, 主模块的同级`package`被视为顶级包(也就是`top-level package`), 所以主模块所在包其实是在python解释器解析到的顶层包之外的, 如果不小心以相对导入的方式引用到了, 就会报`beyond top-level package`这个错误

```md
TestModule/
    ├── main.py # from Tom import tom; print(__name__)
    ├── __init__.py
    ├── Tom
    │   ├── __init__.py # print(__name__)
    │   ├── tom.py # from . import tom_brother; from ..Kate import kate; print(__name__)
    │   └── tom_brother.py # print(__name__) 
    └── Kate      
         ├── __init__.py # print(__name__)
         └── kate.py # print(__name__)
```

- 把`main.py`移动到`TestModule`文件夹外面, 使之与`TestModule`平级, 这样`TestModule`即会被解析器视为一个`package`, 在其他模块中使用相对导入的方式引用到了也不会报错

```md
src/
├── main.py # from TestModule.Tom import tom; print(__name__)
└── TestModule/
        ├── __init__.py # print(__name__)
        ├── Tom
        │   ├── __init__.py # print(__name__)
        │   ├── tom.py # from . import tom_brother; from ..Kate import kate; print(__name__)
        │   └── tom_brother.py # print(__name__) 
        └── Kate      
             ├── __init__.py # print(__name__)
             └── kate.py # print(__name__)
```

- `tom.py`中将`TestModule`包加入到`sys.path`变量中, 并使用绝对导入的方式导入`Kate`包, 修改后的`tom.py`内容如下

```python
from . import tom_brother
import os, sys
sys.path.append("..") # 等价于 sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Kate import kate # 改成绝对导入的方式导入Kate
print(__name__)
```
