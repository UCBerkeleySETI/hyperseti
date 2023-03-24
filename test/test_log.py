from hyperseti.log import get_logger, update_levels, set_log_level

def test_log():
    log = get_logger('hyperseti.dedoppler')

    update_levels('info')

    log.info("Hello")

    update_levels("critical")
    log.critical("Uh-oh!")
    log.info("Won't see this")

    set_log_level("notice")
    log.notice("This is a notice")
    log.info("won't see this though")

if __name__ == "__main__":
    test_log()