import cProfile
import pstats
import io
import logging
from django.utils.deprecation import MiddlewareMixin
import os

logger = logging.getLogger(__name__)

class ProfilingMiddleware(MiddlewareMixin):
    def process_view(self, request, view_func, view_args, view_kwargs):
        self.pr = cProfile.Profile()
        self.pr.enable()
        return None

    def process_response(self, request, response):
        self.pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)

        # Profil sonuçlarını string olarak al
        stats_str = s.getvalue()

        # Detaylı fonksiyon sürelerini loglamak için kendi yazdığımız modülleri filtreleyelim
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        detailed_stats = []
        for func, (cc, nc, tt, ct, callers) in ps.stats.items():
            # Sadece proje dizininde bulunan dosyalara ait fonksiyonları filtrele
            if project_path in func[0]:
                func_name = f"{func[0]}:{func[1]}:{func[2]}"
                detailed_stats.append(
                    f"Fonksiyon: {func_name}, Çağrı sayısı: {cc}, Toplam süre: {tt:.6f} sn, "
                    f"Kümülatif süre: {ct:.6f} sn"
                )

        print("*******************************")
        for i in detailed_stats:
            print(i)
        print("*******************************")

        # Loglama
        logger.info("Profiling sonuçları %s %s için:", request.method, request.path)
        logger.info(stats_str)
        if detailed_stats:
            logger.info("Detaylı fonksiyon süreleri:")
            for stat in detailed_stats:
                logger.info(stat)
        else:
            logger.info("Detaylı fonksiyon süresi bulunamadı.")

        return response
