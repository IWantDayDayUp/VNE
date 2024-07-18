# argparse

## add_argument() 方法

```Python

ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])

- name or flags: 一个命名或者一个选项字符串的列表，例如 foo(位置参数) 或 -f, --foo(可选参数)。
- action: 当参数在命令行中出现时使用的动作基本类型。
- nargs: 命令行参数应当消耗的数目。
- const: 被一些 action 和 nargs 选择所需求的常数。
- default: 当参数未在命令行中出现并且也不存在于命名空间对象时所产生的值。
- type: 命令行参数应当被转换成的类型。
- choices: 由允许作为参数的值组成的序列。
- required: 此命令行选项是否可省略 （仅选项可用）。
- help: 一个此选项作用的简单描述。
- metavar: 在使用方法消息中使用的参数值示例。
- dest: 被添加到 parse_args() 所返回对象上的属性名

```

例子:

```Python
# '--p_net_setting_path': 可选参数
# type=str: 参数类型 - 字符串
# default='settings/p_net_setting_multi_resource.yaml': 默认
# help='File path of physical network settings': 提示信息
data_arg.add_argument('--p_net_setting_path', type=str, default='settings/p_net_setting_multi_resource.yaml', help='File path of physical network settings')

```

## parse_args() 方法

```Python

ArgumentParser.parse_args(args=None, namespace=None)
将参数字符串转换为对象并将其设为命名空间的属性。 返回带有成员的命名空间

- args: 要解析的字符串列表。 默认值是从 sys.argv 获取。
- namespace: 用于获取属性的对象。 默认值是一个新的空 Namespace 对象

```

## 参数组

```Python

# 在默认情况下，ArgumentParser 会在显示帮助消息时将命令行参数分为“位置参数”和“可选参数”两组。 
# 当存在比默认更好的参数分组概念时，可以使用 add_argument_group() 方法来创建适当的分组
ArgumentParser.add_argument_group(title=None, description=None)

```
