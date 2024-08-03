
class NilaiController {
    static getNilaiAkhir(req, res) {
        const { name, score } = req.query;
        if (!name || !score) {
          res.send(400, { message: "Name and score are required" });
          return;
        }
        res.send(200, {
          message: "Nilai akhir berhasil diambil",
          name: name || "Name not provided",
          score: score || "Score not provided"
        });
    }
  
    static postData(req, res) {
        if (!req.body.name || !req.body.score) {
          res.send(400, { message: "Name and score are required" });
          return;
        }
        res.send(201, { message: "Data berhasil dikirim", data: req.body });
    }
  
    static putData(req, res) {
      res.send(200, { message: "Data berhasil diperbarui", data: req.body });
    }
  
    static deleteData(req, res) {
      res.send(200, { message: "Data berhasil dihapus" });
    }
  }
  
  module.exports = NilaiController;
  