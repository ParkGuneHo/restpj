from .models import Essay , Album , Files
from rest_framework import serializers

class EssaySerializer(serializers.ModelSerializer):

    author_name = serializers.ReadOnlyField(source= 'author.username')

    class Meta:
        model = Essay
        fields = ('pk','title','body','author_name')


class AlbumSerializer(serializers.ModelSerializer):

    author_name = serializers.ReadOnlyField(source= 'author.username')
    image = serializers.ImageField(use_url = True) #이미지를 업로드 하고 결과값을 확인 하는 작업을 url로 하겠다.

    class Meta:
        model = Album
        fields = ('pk','author_name','image','desc')

class FilesSerializer(serializers.ModelSerializer):

    author_name = serializers.ReadOnlyField(source= 'author.username')
    files = serializers.FileField(use_url = True)

    class Meta:
        model = Files
        fields = ('pk','author','myfile','desc')