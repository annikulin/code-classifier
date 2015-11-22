from django.http import HttpResponse, Http404
from django.template import RequestContext, loader
from bayes_classifier import BayesTextClassifier
from classifier_trainer import train_classifier

text_classifier = BayesTextClassifier()
train_classifier(text_classifier, 20)


def index(request):
    if request.method == 'POST':
        code_listing = request.POST['code_listing']
        print code_listing
        classified_lang = text_classifier.test(code_listing)
        print classified_lang
        template = loader.get_template('naive_bayes/index.html')
        context = RequestContext(request, {
            "language": classified_lang,
        })
        return HttpResponse(template.render(context))
    else:
        template = loader.get_template('naive_bayes/index.html')
        context = RequestContext(request, {})
        return HttpResponse(template.render(context))
