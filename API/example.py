import api
p=api.Line([api.np.linspace(0,10,100)],[api.np.random.randint(1,10,(100))])
p.set_params("mode","lines")
a=api.np.random.rand((100))
b=api.np.random.rand((100))
c=api.np.linspace(0,20,100)
d=api.np.linspace(0,20,100)
m=[ api.Bar([api.np.linspace(0,10,100)],[api.np.random.randint(1,10,(100))]),
    api.Line([api.np.linspace(0,10,100)],[api.np.random.randint(1,10,(100))]),
    p,
    api.Pie([["a","b","c","d","e"]],[[100,200,300,400,500]]),
    api.Contour([api.np.arange(0,100,1)],[api.np.arange(0,100,1)],[api.np.random.randn(100,100)]),
    api.Histogram2D([a], [b]),
    api.Redar([["a","b","c","d","e","f"]],[[200,300,100,250,450,300]]),
    api.Surface([lambda x,y:api.np.log(x+1)+api.np.log(y+1)],[c],[d]),
    api.Mash3D([lambda x,y:x+y],[api.np.random.rand(100)],[api.np.random.rand(100)],[api.np.random.rand(100)]),
    api.Table([[["a","b","c"],[10,20,30]]])

    ]
api.Subplot(m).show()