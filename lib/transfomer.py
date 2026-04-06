# 인풋 데이터 증강 코드
import albumentations as A


'''
data loading 후 albuumentation을 이용해 데이터 증강을 수행한다.
input 및 output 이미지 모두 동일하게 증강이 이뤄져야 한다. (mask)
'''

def data_Agumentation():
    transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),

    ], is_check_shapes=False)

    return transform
