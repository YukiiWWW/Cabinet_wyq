from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class MyDataset(XMLDataset):

    CLASSES = ('XinmaichaoBread', 'MiduoqiBun', 'TaoliBread', 'ShuanghuiSausage', 'YiliYogurt',
               'YiliMilk_box', 'WahahaAD_220g', 'RedBull_tin', 'BaisuishanWater', 'Lindaer',
               'ZhongtianEgg_2', 'ZhenxiangBeef', 'MasterKongNoodle', 'ShangbaQuailegg', 'OreoBiscuit',
               'JinluoSausage', 'DanengMaidong', 'WatsonsSodaWater_tin', 'YiliFruitYogurt', 'Nescafe_bottle',
               'WangzaiMilk_245ml', 'VitaminTea_box', 'NongfuShuirongC100', 'WeilongDamianjin_102g', 'MasterKongIceBlackTea',
               'WangwangTease', 'NongfuChaPi', 'BoluBiscuits', 'CocaCola_bottle', 'WangwangGrain',)

    def __init__(self, **kwargs):
        super(MyDataset, self).__init__(**kwargs)
        if '0724' in self.img_prefix:
            self.year = 0o0724
        else:
            self.year = 2020
