import grpc
import sybil_pb2_grpc
from server import SybilService
from concurrent import futures

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sybil_pb2_grpc.add_SybilServicer_to_server(SybilService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()