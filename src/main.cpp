#include "ascend_backend.hpp"
#include "data.hpp"
#include "native_math.hpp"
#include "protocol.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <iostream>
#include <memory>
#include <netdb.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
using namespace neb;
std::atomic<bool> stop{false};
void signal_exit(int){stop=true;}
struct Options {
    std::vector<std::string> a;
    bool has(const std::string& key)const{return std::find(a.begin(),a.end(),key)!=a.end();}
    std::string get(const std::string& k,const std::string& fallback="")const{
        for(size_t i=0;i<a.size();++i)if(a[i]==k){if(i+1>=a.size())throw std::runtime_error("missing value for "+k);return a[i+1];}
        return fallback;
    }
    int number(const std::string& k,int fallback)const{return std::stoi(get(k,std::to_string(fallback)));}
    uint64_t u64(const std::string& k,uint64_t fallback)const{return std::stoull(get(k,std::to_string(fallback)));}
};
std::vector<uint8_t> require_hex(const Options&opt,const std::string&k,size_t n,const std::string&fallback=""){
    auto b=unhex(opt.get(k,fallback));if(b.size()!=n)throw std::runtime_error(k+" must be "+std::to_string(n)+" bytes hex");return b;
}
std::unique_ptr<neb2_model> make_model(const std::vector<uint8_t>& packed){
    auto model=std::make_unique<neb2_model>();
    if(an_set_model(model.get(),packed.data(),packed.size())!=0)
        throw std::runtime_error("model requires 13000 packed signed-INT4 bytes");
    return model;
}
class Scanner {
public:
  Scanner(AscendDevice& device,const DataSet& dataset,const neb2_model& model,int threads,int audits):
    device_(device),dataset_(dataset),model_(model),audits_(audits) {
    auto t=std::chrono::steady_clock::now();
    std::cout<<"[table] building "<<dataset_.n<<" sample gradients (reference C)..."<<std::endl;
    table_=build_table(model_,dataset_,threads);
    device_.upload(table_.conv,table_.factors,table_.n);
    const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t).count();
    std::cout<<"[table] ready, NPU GM bytes="
       <<table_.conv.size()*sizeof(int16_t)+table_.factors.size()
       <<", prep+upload="<<ms<<"ms"<<std::endl;
  }
  struct Hit {uint32_t nonce;bool block;std::array<uint8_t,32> ticket;};
  std::vector<Hit> scan(const uint8_t header[80],uint32_t start,uint32_t count,
      const uint8_t share_target[32],const uint8_t block_target[32]) {
    if(count==0)return {};
    std::vector<uint8_t> keys(static_cast<size_t>(count)*32),digests(static_cast<size_t>(count)*32),tickets(static_cast<size_t>(count)*32);
    std::vector<uint32_t> indices(static_cast<size_t>(count)*16);
    an_batch_keys_indices(header,start,count,dataset_.n,keys.data(),indices.data());
    auto got=device_.run(indices);
    for(uint32_t i=0;i<count;++i)to_bytes(got[i],digests.data()+static_cast<size_t>(i)*32);
    an_batch_tickets(digests.data(),keys.data(),count,tickets.data());
    std::vector<Hit> hits;
    for(uint32_t i=0;i<count;++i){
      const uint32_t nonce=start+i;
      const auto* ticket=tickets.data()+static_cast<size_t>(i)*32;
      const bool share=meets_target(ticket,share_target);
      const bool block=meets_target(ticket,block_target);
      const bool audit=(verified_<static_cast<uint64_t>(std::max(0,audits_)) || nonce%4096u==0u || share || block);
      if(audit){
        uint8_t reference_digest[32]={},reference_ticket[32]={};
        const int rc=an_reference_eval(&model_,dataset_.values.data(),dataset_.labels.data(),dataset_.n,
           keys.data()+static_cast<size_t>(i)*32,reference_digest,reference_ticket);
        if(rc!=0||memcmp(reference_digest,digests.data()+static_cast<size_t>(i)*32,32)!=0||
            memcmp(reference_ticket,ticket,32)!=0){
            throw std::runtime_error("CONSENSUS MISMATCH nonce="+std::to_string(nonce)+"; refused to submit");
        }
        ++verified_;
      }
      if(share||block){
        Hit hit{nonce,block,{}};memcpy(hit.ticket.data(),ticket,32);hits.push_back(hit);
        std::cout<<"[hit] nonce="<<nonce<<" block="<<block<<" ticket="<<hex(ticket,32)<<std::endl;
      }
    }
    attempted_+=count;
    return hits;
  }
  uint64_t attempted()const{return attempted_;}
  uint64_t verified()const{return verified_;}
private:
  AscendDevice&device_;const DataSet&dataset_;const neb2_model&model_;
  GradientTable table_;
  int audits_=3;uint64_t attempted_=0,verified_=0;
};
void print_help(){
  std::cout<<"ascend_neb: native Ascend C KEP1 NEB scanner; zero PyTorch dependencies\n"
    <<"Usage:\n"
    <<"  ascend_neb selftest [--device 0] [--rows 48] [--threads 4]\n"
    <<"  ascend_neb scan --dataset neb-datasets.bin --model packed.bin --header-hex 160hex\n"
    <<"      [--target-hex 64hex] [--count 16] [--start 0] [--device 0] [--batch 4]\n"
    <<"  ascend_neb mine --dataset neb-datasets.bin [--node 127.0.0.1:3441] [--device 0]\n"
    <<"      [--batch 4] [--audit 3] [--threads 4] [--max-attempts 0]\n"
    <<"      [--partition-index 0] [--partitions 1]\n"
    <<"Options: --allow-other-dataset (FOR LAB DATA ONLY; disables pinned dataset hash)\n";
}
void run_selftest(const Options&opt){
  const int dev=opt.number("--device",0),n=opt.number("--rows",48);
  if(n<1||n>100000)throw std::runtime_error("rows must be 1..100000");
  DataSet ds;ds.arch=1;ds.n=static_cast<uint32_t>(n);
  ds.values.resize(static_cast<size_t>(n)*784);ds.labels.resize(n);
  for(int r=0;r<n;++r){ds.labels[r]=uint8_t(r%2);for(int j=0;j<784;++j)
    ds.values[static_cast<size_t>(r)*784+j]=uint8_t((r*13+j*7+(j/17))%8);}
  std::vector<uint8_t> packed(13000);
  for(size_t i=0;i<packed.size();++i)packed[i]=uint8_t((i*71+17)%256);
  auto model=make_model(packed);
  AscendDevice device(dev);
  Scanner scanner(device,ds,*model,opt.number("--threads",0),8);
  uint8_t hdr[80]={};for(int i=0;i<80;++i)hdr[i]=uint8_t((i*7+5)%256);
  std::array<uint8_t,32> target;target.fill(0xff);
  scanner.scan(hdr,13,static_cast<uint32_t>(opt.number("--attempts",3)),target.data(),target.data());
  std::cout<<"[PASS] Ascend C digest/Ticket match official NEB C reference; verified="<<scanner.verified()<<"\n";
}
void run_scan(const Options&opt){
  const std::string dsfile=opt.get("--dataset"),modelpath=opt.get("--model");
  if(dsfile.empty()||modelpath.empty())throw std::runtime_error("--dataset and --model required");
  auto datasets=parse_datasets(read_file(dsfile),opt.has("--allow-other-dataset"));
  const int slot=opt.number("--slot",0);
  if(slot<0||slot>=static_cast<int>(datasets.size()))throw std::runtime_error("bad dataset slot");
  auto model=make_model(read_file(modelpath));
  auto header=require_hex(opt,"--header-hex",80);
  const auto target=require_hex(opt,"--target-hex",32,std::string(64,'0'));
  const uint64_t start=opt.u64("--start",0),count=opt.u64("--count",16);
  if(start>0xffffffffull||count>0x100000000ull-start)
    throw std::runtime_error("nonce range overflow");
  const uint32_t batch=uint32_t(opt.number("--batch",64));
  if(batch<1||batch>4096)throw std::runtime_error("batch must be 1..4096");
  AscendDevice device(opt.number("--device",0));
  Scanner scanner(device,datasets[slot],*model,opt.number("--threads",0),opt.number("--audit",3));
  auto t=std::chrono::steady_clock::now();uint64_t offset=0,total_hits=0;
  while(offset<count){
    const uint32_t n=static_cast<uint32_t>(std::min<uint64_t>(batch,count-offset));
    auto hits=scanner.scan(header.data(),uint32_t(start+offset),n,target.data(),target.data());
    total_hits+=hits.size();offset+=n;
  }
  const double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  std::cout<<"[scan complete] attempts="<<count<<" audited="<<scanner.verified()
    <<" hits="<<total_hits<<" scan_s="<<s<<" rate="<<count/std::max(1e-9,s)<<"/s\n";
}
struct Conn {
    int fd=-1;
    explicit Conn(const std::string&address){
        auto colon=address.rfind(':');if(colon==std::string::npos)throw std::runtime_error("node must be HOST:PORT");
        std::string hostname=address.substr(0,colon),port=address.substr(colon+1);
        addrinfo hint{};hint.ai_socktype=SOCK_STREAM;hint.ai_family=AF_UNSPEC;
        addrinfo *list=nullptr;
        int rc=getaddrinfo(hostname.c_str(),port.c_str(),&hint,&list);
        if(rc)throw std::runtime_error("getaddrinfo: "+std::string(gai_strerror(rc)));
        for(addrinfo *p=list;p;p=p->ai_next){
            int sock=socket(p->ai_family,p->ai_socktype,p->ai_protocol);
            if(sock<0)continue;
            if(connect(sock,p->ai_addr,p->ai_addrlen)==0){fd=sock;break;}
            close(sock);
        }
        freeaddrinfo(list);
        if(fd<0)throw std::runtime_error("node connect failed: "+address);
    }
    ~Conn(){if(fd>=0)close(fd);}
    void send(const std::vector<uint8_t>& message){
        size_t off=0;while(off<message.size()){
            ssize_t r=::send(fd,message.data()+off,message.size()-off,MSG_NOSIGNAL);
            if(r<=0)throw std::runtime_error("node send failed");
            off+=size_t(r);
        }
    }
    void read(Framer& frames,int timeout_ms){
        pollfd pfd{fd,POLLIN,0};const int r=poll(&pfd,1,timeout_ms);
        if(r<0)throw std::runtime_error("node poll failed");
        if(r==0)return;
        if(pfd.revents&(POLLERR|POLLHUP|POLLNVAL))throw std::runtime_error("node closed");
        if(pfd.revents&POLLIN){uint8_t b[65536];ssize_t n=recv(fd,b,sizeof b,0);
            if(n<=0)throw std::runtime_error("node disconnected");
            frames.append(b,size_t(n));}
    }
};
void run_mine(const Options&opt){
  const auto dataset_path=opt.get("--dataset");if(dataset_path.empty())throw std::runtime_error("--dataset required");
  auto datasets=parse_datasets(read_file(dataset_path),opt.has("--allow-other-dataset"));
  const uint64_t partitions=opt.u64("--partitions",1),part=opt.u64("--partition-index",0);
  if(partitions<1||partitions>65536||part>=partitions)throw std::runtime_error("bad nonce partition");
  const uint64_t lower=(0x100000000ull*part)/partitions;
  const uint64_t upper=(0x100000000ull*(part+1))/partitions;
  const uint32_t batch=uint32_t(opt.number("--batch",64));
  if(batch<1||batch>4096)throw std::runtime_error("batch must be 1..4096");
  const int threads=opt.number("--threads",0),audit=opt.number("--audit",3);
  const uint64_t max_attempts=opt.u64("--max-attempts",0);
  AscendDevice device(opt.number("--device",0));
  Conn node(opt.get("--node","127.0.0.1:3441"));node.send(hello("ascend_neb-native/0.2"));
  std::cout<<"[node] connected; waiting for model and work\n";
  std::unique_ptr<neb2_model> model;
  std::unique_ptr<Scanner> scanner;
  std::array<uint8_t,32> loaded_hash{};
  uint8_t active_slot=0;
  std::unique_ptr<Job> job;
  Framer frames;
  uint64_t nonce=lower,total=0;
  auto then=std::chrono::steady_clock::now();
  while(!stop && (max_attempts==0||total<max_attempts)){
    node.read(frames,job&&model?0:2000);
    Frame msg;
    while(frames.pop(msg)){
      if(msg.kind==7){
        if(msg.payload.size()!=13033)throw std::runtime_error("invalid MODEL payload length");
        uint8_t slot=msg.payload[0];if(slot>=datasets.size())throw std::runtime_error("MODEL slot exceeds datasets");
        auto next=make_model(std::vector<uint8_t>(msg.payload.begin()+33,msg.payload.end()));
        uint8_t real_hash[32]={};an_weights_hash(next.get(),real_hash);
        if(memcmp(real_hash,msg.payload.data()+1,32)!=0)throw std::runtime_error("MODEL weight hash mismatch");
        if(!model||active_slot!=slot||memcmp(real_hash,loaded_hash.data(),32)!=0){
          active_slot=slot;memcpy(loaded_hash.data(),real_hash,32);model=std::move(next);scanner.reset();
          job.reset();
          std::cout<<"[model] slot="<<unsigned(slot)<<" weights="<<hex(real_hash,8)<<"...\n";
        }
      } else if(msg.kind==2){
        job=std::make_unique<Job>(Job::decode(msg.payload));nonce=lower;
        std::cout<<"[job] height="<<job->height<<" id="<<job->id<<" expiry="<<job->expiry<<"\n";
      } else if(msg.kind==5){
        std::cout<<"[node result] accepted="<<(msg.payload.empty()?0:msg.payload[0])<<"\n";
      }
    }
    if(!model||!job)continue;
    if(job->expiry>0&&static_cast<uint64_t>(time(nullptr))>=job->expiry){job.reset();continue;}
    if(nonce>=upper){job.reset();continue;}
    if(!scanner){scanner=std::make_unique<Scanner>(device,datasets[active_slot],*model,threads,audit);}
    uint64_t n=std::min<uint64_t>(batch,upper-nonce);
    if(max_attempts)n=std::min<uint64_t>(n,max_attempts-total);
    if(n==0)break;
    auto header=job->header();
    auto hits=scanner->scan(header.data(),uint32_t(nonce),static_cast<uint32_t>(n),
                   job->share_target.data(),job->target.data());
    for(const auto& h:hits){
      if(h.block)node.send(submit(job->id,std::vector<uint8_t>(job->extra_len,0),job->time_start,h.nonce));
      else node.send(share(job->id,h.nonce));
      std::cout<<"[sent] "<<(h.block?"block":"share")<<" nonce="<<h.nonce<<"\n";
    }
    nonce+=n;total+=n;
    if(total%std::max<uint64_t>(512,batch)==0){
      double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-then).count();
      std::cout<<"[rate] total="<<total<<" attempts/s="<<(total/std::max(1e-9,seconds))
        <<" verified="<<scanner->verified()<<"\n";
    }
  }
  std::cout<<"[miner] stopped, attempts="<<total<<"\n";
}
} // namespace
int main(int argc,char**argv){
  try{
    if(argc<2||std::string(argv[1])=="--help"||std::string(argv[1])=="help"){print_help();return 0;}
    std::signal(SIGINT,signal_exit);std::signal(SIGTERM,signal_exit);
    Options opt;for(int i=2;i<argc;++i)opt.a.emplace_back(argv[i]);
    const std::string cmd=argv[1];
    if(cmd=="selftest")run_selftest(opt);
    else if(cmd=="scan")run_scan(opt);
    else if(cmd=="mine")run_mine(opt);
    else throw std::runtime_error("unknown command: "+cmd);
    return 0;
  }catch(const std::exception&e){std::cerr<<"[ERROR] "<<e.what()<<"\n";return 2;}
}
