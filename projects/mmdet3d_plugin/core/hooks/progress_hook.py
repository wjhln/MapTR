from mmcv.runner import HOOKS, Hook
import mmcv


@HOOKS.register_module()
class CustomProgressBarHook(Hook):
    """自定义进度条Hook，用于显示训练过程中的迭代进度"""

    def __init__(self, print_interval=10):
        self.print_interval = print_interval
        self.prog_bar = None

    def before_train_epoch(self, runner):
        """在每个epoch开始前初始化进度条"""
        if runner.rank == 0:
            # 获取当前epoch的数据加载器长度
            data_loader = runner.data_loader
            self.prog_bar = mmcv.ProgressBar(len(data_loader))

    def after_train_iter(self, runner):
        """在每个迭代后更新进度条"""
        if runner.rank == 0 and self.prog_bar is not None:
            # 每隔一定间隔更新进度条
            if runner.inner_iter % self.print_interval == 0 or runner.inner_iter == len(runner.data_loader) - 1:
                self.prog_bar.update()

    def after_train_epoch(self, runner):
        """在每个epoch结束后清理进度条"""
        if runner.rank == 0 and self.prog_bar is not None:
            # 确保进度条完成
            remaining = len(runner.data_loader) - self.prog_bar.completed
            for _ in range(remaining):
                self.prog_bar.update()
            print()  # 换行
