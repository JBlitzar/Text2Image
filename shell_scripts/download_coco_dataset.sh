read -p "Continue? This is a ~25gb download (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    mkdir data
    cd data
    curl -o train_imgs.zip http://images.cocodataset.org/zips/train2017.zip
    curl -o val_imgs.zip http://images.cocodataset.org/zips/val2017.zip
    curl -o test_imgs.zip http://images.cocodataset.org/zips/test2017.zip


    curl -o annotations_trainval.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi

