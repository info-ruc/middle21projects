from django.shortcuts import render

# Create your views here.
from django.shortcuts import HttpResponse,redirect  # 导入HttpResponse模块
import user.dataquery as dq


def index(request):  # request是必须带的实例。类似class下方法必须带self一样
    error_msg = ''

    if request.method == "POST":

        chose = request.POST.get('chose', None)  # 避免提交空，时异常

        query = request.POST.get('query', None)

        if chose in ['1','2','3'] and query != '':

            print('chose=' + chose, 'query=' + query)
            result = dq.queries(chose, query)
            return render(request, 'index.html', {'results': result})

    return render(request, 'index.html')
