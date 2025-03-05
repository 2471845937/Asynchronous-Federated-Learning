import socket
import pickle
import torch
import threading
import time
import matplotlib.pyplot as plt
from model import create_model, test
from data_loader import get_datasets


class FederatedServer:
    def __init__(self):
        self.model = create_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.global_round = 0
        self.accuracies = []
        self.train_losses = []
        self.train_accuracies = []
        self.best_accuracy = 0.0

        # 数据集和测试加载器
        _, test_data = get_datasets(0)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

        # 客户端管理
        self.client_sockets = {}
        self.client_sample_counts = {}
        self.pending_clients = []
        self.lock = threading.Lock()

    def aggregate(self, client_params, weights):
        global_dict = self.model.state_dict()
        for key in global_dict:
            client_tensors = [param[key].float() for param in client_params]
            weighted_tensors = [tensor * weight for tensor, weight in zip(client_tensors, weights)]
            global_dict[key] = sum(weighted_tensors)
        self.model.load_state_dict(global_dict)

    def handle_client(self, client_socket, addr):
        while True:
            try:
                # 接收数据长度
                data_len_bytes = client_socket.recv(4)
                if not data_len_bytes:
                    raise ConnectionError("No data received")
                data_len = int.from_bytes(data_len_bytes, 'big')

                # 接收数据
                data = b''
                while len(data) < data_len:
                    packet = client_socket.recv(data_len - len(data))
                    if not packet:
                        raise ConnectionError("Incomplete data")
                    data += packet

                update = pickle.loads(data)
                self.client_sample_counts[addr] = update['sample_count']

                # 计算权重并聚合
                sample_counts = list(self.client_sample_counts.values())
                total_samples = sum(sample_counts)
                weights = [count / total_samples for count in sample_counts]

                # 聚合参数
                self.aggregate(
                    [update['params'] for update in [update]],
                    weights
                )

                # 记录训练指标
                avg_loss = update['loss']
                avg_acc = update['accuracy']
                self.train_losses.append(avg_loss)
                self.train_accuracies.append(avg_acc)

                # 测试全局模型
                test_loss, test_acc = test(self.model, self.test_loader, self.device)
                self.accuracies.append(test_acc)

                # 保存最佳模型
                if test_acc > self.best_accuracy:
                    self.best_accuracy = test_acc
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    print(f"New best model saved with accuracy {self.best_accuracy:.2f}%")

                # 打印结果
                print(f"\nGlobal Round {self.global_round} Results:")
                print(f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.2f}%")
                print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

                self.global_round += 1

            except Exception as e:
                print(f"Error receiving from {addr}: {e}")
                with self.lock:
                    if addr in self.client_sockets:
                        del self.client_sockets[addr]
                        del self.client_sample_counts[addr]
                break

        client_socket.close()
        print(f"Client {addr} disconnected.")

    def run(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('192.168.1.203', 12345))
        server_socket.listen(5)

        # 客户端接受线程
        def accept_clients():
            while True:
                try:
                    client_socket, addr = server_socket.accept()
                    print(f"\nNew client connected: {addr}")
                    with self.lock:
                        self.pending_clients.append((client_socket, addr))
                except Exception as e:
                    if server_socket.fileno() == -1:
                        break  # Server socket closed
                    print(f"Error accepting client: {e}")

        accept_thread = threading.Thread(target=accept_clients, daemon=True)
        accept_thread.start()

        print("Server is running. Press Ctrl+C to stop.")

        try:
            while True:
                # 处理新客户端
                with self.lock:
                    new_clients = self.pending_clients.copy()
                    self.pending_clients.clear()

                # 自动分配任务给新客户端
                for client_socket, addr in new_clients:
                    self.client_sockets[addr] = client_socket
                    self.client_sample_counts[addr] = 0
                    print(f"Client {addr} added to training pool.")
                    threading.Thread(target=self.handle_client, args=(client_socket, addr), daemon=True).start()

                # 如果有客户端，开始训练
                if self.client_sockets:
                    print(f"\nStarting Global Round {self.global_round} with {len(self.client_sockets)} clients.")

                    # 发送训练指令和全局模型
                    model_data = pickle.dumps(self.model.state_dict())
                    model_size = len(model_data)
                    disconnected = []

                    # 发送训练指令和模型
                    for addr in list(self.client_sockets.keys()):
                        sock = self.client_sockets[addr]
                        try:
                            sock.sendall(b'TRAIN')
                            sock.sendall(model_size.to_bytes(4, 'big'))
                            sock.sendall(model_data)
                        except Exception as e:
                            print(f"Error sending to {addr}: {e}")
                            disconnected.append(addr)

                    # 移除断开连接的客户端
                    for addr in disconnected:
                        del self.client_sockets[addr]
                        del self.client_sample_counts[addr]
                    if disconnected:
                        print(f"Removed disconnected clients: {disconnected}")

                else:
                    print("No clients in training pool. Waiting...")
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\nShutting down server...")
            for sock in self.client_sockets.values():
                try:
                    sock.sendall(b'EXIT')
                    sock.close()
                except:
                    pass
            server_socket.close()
            self.plot_performance()

    def plot_performance(self):
        if not self.global_round:
            print("No training data to plot.")
            return

        plt.figure(figsize=(12, 5))

        # 确保数据长度一致
        rounds = range(1, self.global_round + 1)
        plt.subplot(1, 2, 1)
        plt.plot(rounds, self.accuracies[:self.global_round], marker='o', label='Test')
        plt.plot(rounds, self.train_accuracies[:self.global_round], marker='s', label='Train')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(rounds, self.train_losses[:self.global_round], marker='o', color='orange')
        plt.xlabel('Communication Round')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('fl_performance.png')
        plt.close()
        print("Performance plot saved to fl_performance.png")


if __name__ == "__main__":
    server = FederatedServer()
    server.run()