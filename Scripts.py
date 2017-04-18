import timeit
import cPickle

start = timeit.timeit()
print "hello"
tt = cPickle.load(file("keras_frcnn/Data/VisualGenome/final_entities.p",'rb'))

end = timeit.timeit()
print end - start