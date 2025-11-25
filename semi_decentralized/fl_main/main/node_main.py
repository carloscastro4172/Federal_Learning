# fl_main/main/node_main.py
import asyncio
import logging

from fl_main.lib.util.helpers import (
    generate_id,
    get_ip,
    set_config_file,
    read_config,
)
from fl_main.lib.util.communication_handler import send_websocket, receive

import websockets  # asegúrate de tener websockets instalado


async def get_role_and_model(init_ip: str, init_port: int):
    """
    Se conecta al InitServer, se registra y pide rol + modelo.
    Si el InitServer aún no ha elegido agregador, reintenta cada 2s.
    """
    my_id = generate_id()
    my_ip = get_ip()

    uri = f"ws://{init_ip}:{init_port}"
    logging.info(f"[NODE] Mi ID = {my_id}, IP = {my_ip}")
    logging.info(f"[NODE] Conectando a InitServer en {uri}")

    while True:
        try:
            async with websockets.connect(uri) as ws:
                # 1) Registrar nodo
                reg_msg = {
                    "msg_type": "register",
                    "component_id": my_id,
                    "ip": my_ip,
                    "port": 0,
                }
                await send_websocket(reg_msg, ws)
                reg_reply = await receive(ws)
                logging.info(f"[NODE] Respuesta registro: {reg_reply}")

                # 2) Pedir rol + modelo
                req = {
                    "msg_type": "get_role_and_model",
                    "component_id": my_id,
                }
                await send_websocket(req, ws)
                reply = await receive(ws)
                logging.info(f"[NODE] Respuesta get_role_and_model: {reply}")

                status = reply.get("status")

                if status == "pending":
                    logging.info(
                        "[NODE] Aún no hay agregador elegido (pending). "
                        "Reintentando en 2 segundos..."
                    )
                    await asyncio.sleep(2)
                    continue

                return reply, my_id

        except Exception as e:
            logging.error(f"[NODE] Error conectando con InitServer: {e}")
            logging.info("[NODE] Reintentando conexión en 2 segundos...")
            await asyncio.sleep(2)


def start_aggregator():
    """
    Arranca el servidor federado (aggregator actual).
    """
    from fl_main.aggregator.server_th import Server
    from fl_main.lib.util.communication_handler import init_fl_server

    s = Server()
    logging.info("[NODE] Soy agregador, iniciando Server()")

    # Crear loop (para que init_fl_server use asyncio.get_event_loop())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # OJO: aquí pasamos la CORUTINA, no la función
    init_fl_server(
        s.register,
        s.receive_msg_from_agent,
        s.model_synthesis_routine(),  # <-- AHORA SÍ: corutina
        s.aggr_ip,
        s.reg_socket,
        s.recv_socket,
    )


def start_client(aggregator_info, model_dict):
    """
    Arranca el cliente federado.
    """
    from fl_main.agent.client import Client

    cl = Client(initial_model=model_dict)

    # IP del agregador (asignada por el InitServer)
    cl.aggr_ip = aggregator_info["ip"]

    logging.info(
        f"[NODE] Soy cliente, mi agregador está en {cl.aggr_ip}:{cl.reg_socket}"
    )

    # Crear loop para que client use asyncio.get_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    cl.start_fl_client()


def main():
    logging.basicConfig(level=logging.INFO)

    # leer config InitServer
    init_cfg_file = set_config_file("init")
    init_cfg = read_config(init_cfg_file)
    init_ip = init_cfg["init_ip"]
    init_port = init_cfg["init_socket"]

    print(init_ip)
    print(init_port)

    # obtener rol y modelo (único sitio donde usamos asyncio.run)
    reply, my_id = asyncio.run(get_role_and_model(init_ip, init_port))

    if reply.get("status") != "ok":
        logging.error(f"Can not obtain the role: {reply}")
        return

    role = reply["role"]
    aggregator_info = reply["aggregator"]
    model_dict = reply["model"]

    logging.info(f"[NODE] Rol recibido: {role}")
    logging.info(f"[NODE] Info agregador: {aggregator_info}")

    if role == "aggregator":
        start_aggregator()
    else:
        start_client(aggregator_info, model_dict)


if __name__ == "__main__":
    main()
