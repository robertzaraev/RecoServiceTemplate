from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
import pandas as pd

from service.api.exceptions import (
    CredentialError,
    ModelNotFoundError,
    UserNotFoundError,
)
from service.api.percentage import get_percentage
from service.log import app_logger

from .config import config_env
from .models import NotFoundError, RecoResponse, UnauthorizedError, \
    ExplainResponse
from .models_zoo import DumpModel, Popular, KNNModelWithTop

import sentry_sdk

router = APIRouter()

api_query = APIKeyQuery(name=config_env["API_KEY_NAME"], auto_error=False)
api_header = APIKeyHeader(name=config_env["API_KEY_NAME"], auto_error=False)
token_bearer = HTTPBearer(auto_error=False)

# sentry_sdk.init(
#     dsn="https://687935e2f3184212a6eeea6eb4784c7d@o4504984015798272.ingest"
#         + ".sentry.io/4504984017305600",
#     traces_sample_rate=1.0,
# )

try:
    models_zoo = {
        "model_1": DumpModel(),
        "popular": Popular(),
        "bm25": KNNModelWithTop(path_to_reco='data/KNNBM25all.csv')
    }
except FileNotFoundError:
    models_zoo = {
        "model_1": DumpModel(),
        "popular": Popular(),
    }


async def get_api_key(
    api_key_query: str = Security(api_query),
    api_key_header: str = Security(api_header),
    token: HTTPAuthorizationCredentials = Security(token_bearer),
):
    if api_key_query == config_env["API_KEY"]:
        return api_key_query
    if api_key_header == config_env["API_KEY"]:
        return api_key_header
    if token is not None and token.credentials == config_env["API_KEY"]:
        return token.credentials
    sentry_sdk.capture_exception(error='Token Error')
    raise CredentialError()


@router.get(path="/health", tags=["Health"])
async def health(api_key: APIKey = Depends(get_api_key)) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        404: {"model": NotFoundError, "user": NotFoundError},
        401: {"model": UnauthorizedError},
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    api_key: APIKey = Depends(get_api_key),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10 ** 6:
        # sentry_sdk.capture_exception(
        #     error=f"User {user_id} not found")
        raise UserNotFoundError(
            error_message=f"User {user_id} not found")

    if user_id % 666 == 0:
        # sentry_sdk.capture_exception(
        #     error=f"User id {user_id} is divided entirely into 666")
        raise UserNotFoundError(
            error_message=f"User id {user_id} is divided entirely into 666")

    if model_name not in models_zoo.keys():
        # sentry_sdk.capture_exception(
        #     error=f"Model {model_name} not found")
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    k_recs = request.app.state.k_recs

    reco = models_zoo[model_name].reco_predict(
        user_id=user_id,
        k_recs=k_recs
    )

    return RecoResponse(user_id=user_id, items=reco)


@router.get(
    path="/explain/{model_name}/{user_id}/{item_id}",
    tags=["Explanations"],
    # response_model=ExplainResponse,
)
async def explain(request: Request, model_name: str, user_id: int,
                  item_id: int) -> ExplainResponse:
    app_logger.info(
        f"Explain for model: {model_name}, user_id: {user_id}, item_id: {item_id}")
    """
    Пользователь переходит на карточку контента, на которой нужно показать
    процент релевантности этого контента зашедшему пользователю,
    а также текстовое объяснение почему ему может понравится этот контент.

    :param request: запрос.
    :param model_name: название модели, для которой нужно получить объяснения.
    :param user_id: id пользователя, для которого нужны объяснения.
    :param item_id: id контента, для которого нужны объяснения.
    :return: Response со значением процента релевантности и текстовым объяснением, понятным пользователю.
    - "p": "процент релевантности контента item_id для пользователя user_id"
    - "explanation": "текстовое объяснение почему рекомендован item_id"
    """
    if user_id > 10 ** 6:
        raise UserNotFoundError(
            error_message=f"User {user_id} not found")

    if user_id % 666 == 0:
        raise UserNotFoundError(
            error_message=f"User id {user_id} is divided entirely into 666")

    if model_name not in models_zoo.keys():
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    k_recs = request.app.state.k_recs

    reco = models_zoo[model_name].reco_predict(
        user_id=user_id,
        k_recs=k_recs
    )
    cold_users = pd.read_csv('data/cold_user_ids.csv')
    if item_id not in reco:
        return ExplainResponse(p=0, explanation='No item in recomendations')
    elif user_id in list(cold_users['user_id']):
        return ExplainResponse(p=0, explanation='Its cold user, just popular')
    else:
        percent = get_percentage(user_id, reco)
        return ExplainResponse(p=percent,
                               explanation=f'The genres of the films you have watched match the recommendations by {percent}%')


def add_views(app: FastAPI) -> None:
    app.include_router(router)
