import socket
import pickle
import torch
import threading
import time
import matplotlib.pyplot as plt
from model import create_model, test
from data_loader import get_datasets
import queue


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
        self.update_queue = queue.Queue()  # 线程安全的队列用于存储客户端更新

    def aggregate(self, client_params, alpha):
        """使用加权平均更新全局模型"""
        global_dict = self.model.state_dict()
        for key in global_dict:
            # 确保张量在正确的设备上并进行类型转换
            global_tensor = global_dict[key].float().to(self.device)
            client_tensor = client_params[key].float().to(self.device)

            # 执行加权平均
            global_dict[key] = (1 - alpha) * global_tensor + alpha * client_tensor

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

                # 将更新放入队列
                with self.lock:
                    self.client_sample_counts[addr] = update['sample_count']
                self.update_queue.put((update['params'], addr))

                print(f"Received update from {addr}")

            except Exception as e:
                print(f"Error receiving from {addr}: {e}")
                with self.lock:
                    if addr in self.client_sockets:
                        del self.client_sockets[addr]
                    if addr in self.client_sample_counts:
                        del self.client_sample_counts[addr]
                break

        client_socket.close()
        print(f"Client {addr} disconnected.")

    def async_aggregation(self):
        """异步聚合客户端更新"""
        while True:
            try:
                # 从队列中获取客户端更新
                client_params, addr = self.update_queue.get()

                # 计算聚合权重
                with self.lock:
                    total_samples = sum(self.client_sample_counts.values())
                    if total_samples == 0:
                        alpha = 0.0
                    else:
                        alpha = self.client_sample_counts[addr] / total_samples

                # 聚合参数
                self.aggregate(client_params, alpha)

                # 记录训练指标
                with self.lock:
                    avg_loss = client_params.get('loss', 0.0)
                    avg_acc = client_params.get('accuracy', 0.0)
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
                print(f"Error during aggregation: {e}")

    def run(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('192.168.1.206', 12345))
        server_socket.listen(5)

        # 启动异步聚合线程
        aggregation_thread = threading.Thread(target=self.async_aggregation, daemon=True)
        aggregation_thread.start()

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

                # 初始化新客户端
                for client_socket, addr in new_clients:
                    with self.lock:
                        self.client_sockets[addr] = client_socket
                        # 初始化样本数为0，直到收到第一次更新
                        self.client_sample_counts[addr] = 0
                    print(f"Client {addr} added to training pool.")
                    threading.Thread(target=self.handle_client, args=(client_socket, addr), daemon=True).start()

                    # 立即发送训练指令
                    try:
                        model_data = pickle.dumps(self.model.state_dict())
                        model_size = len(model_data)
                        client_socket.sendall(b'TRAIN')
                        client_socket.sendall(model_size.to_bytes(4, 'big'))
                        client_socket.sendall(model_data)
                        print(f"Sent training instruction to {addr}")
                    except Exception as e:
                        print(f"Error sending training instruction to {addr}: {e}")

                time.sleep(5)  # 控制轮次频率

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
