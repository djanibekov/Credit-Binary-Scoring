from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from services import predict, pre_process


router = APIRouter()

templates = Jinja2Templates(directory='templates')


@router.get('/', response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})


@router.post("/")
def form_post(request: Request,
              birthday: str = Form(...),
              gender: str = Form(...),
              marital_status: str = Form(...)
              ):
    birth_date = pre_process.calculate_age_by_days(birthday)
    status_arr = pre_process.marital_status_arr(int(marital_status))
    gender_dum = pre_process.gender_arr(int(gender))

    X = gender_dum + status_arr + birth_date
    result = predict.MakePrediction()

    if result.predict(X):
        decision = 'OK'
    else:
        decision = 'Bad'

    return templates.TemplateResponse('index.html', context={'request': request, 'result': decision, 'data': X})
